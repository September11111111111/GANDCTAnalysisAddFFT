"""
app.py — GAN 检测 Web 应用 (Streamlit)
=======================================
启动:  streamlit run app.py

功能:
  - 单张检测: 上传 jpg/png → 显示原图 + 热力图 + 判定 + 置信度
  - 批量检测: 上传 zip → 自动过滤非图片 → 进度条 → 结果表格 + CSV 下载
  - 历史记录: SQLite 存储, 缩略图 + 4×4 热力图数值 (不存原图)
  - 模型设置: SimpleCNN / MIL 切换

依赖:
  pip install streamlit tensorflow==2.13.1 pandas pillow numpy scipy scikit-learn matplotlib joblib
"""

import os
import io
import zipfile
import sqlite3
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import inference

# ─────────────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────────────

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_BASE_DIR, "inference_history.db")

# 论文截图用的示例样本目录（来自训练集 most_fake / most_real）
SAMPLE_FAKE_DIR = os.path.join(_BASE_DIR, "runs", "stylegan_mil_full",
                               "heatmaps", "most_fake")
SAMPLE_REAL_DIR = os.path.join(_BASE_DIR, "runs", "stylegan_mil_full",
                               "heatmaps", "most_real")

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ZIP_MAX_IMAGES     = 1000          # zip 内最多处理多少张
SINGLE_MAX_BYTES   = 50 * 1024 * 1024   # 单文件最大 50 MB


def _safe_rerun():
    """兼容 Streamlit < 1.27 (无 st.rerun, 用 st.experimental_rerun)."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def _list_sample_image_paths(category, k=8):
    """category ∈ {'fake', 'real'}; 返回前 k 张样本路径（按文件名排序）."""
    d = SAMPLE_FAKE_DIR if category == "fake" else SAMPLE_REAL_DIR
    if not os.path.isdir(d):
        return []
    files = sorted(f for f in os.listdir(d)
                   if os.path.splitext(f)[1].lower() in ALLOWED_IMAGE_EXTS)
    return [os.path.join(d, f) for f in files[:k]]


def _build_sample_zip(n_fake=10, n_real=10):
    """合成内存 ZIP（fake/ + real/ 两个子目录），用于批量页一键演示."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in _list_sample_image_paths("fake", n_fake):
            zf.write(p, arcname=f"fake/{os.path.basename(p)}")
        for p in _list_sample_image_paths("real", n_real):
            zf.write(p, arcname=f"real/{os.path.basename(p)}")
    return buf.getvalue()

st.set_page_config(
    page_title="GAN Image Detector",
    page_icon="🔍",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# SQLite 历史记录
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    """创建表 (如果不存在). 字段:
       id, timestamp, filename, label, score, route,
       arch, latency_ms, thumbnail_png(BLOB), heatmap_json(TEXT) """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT    NOT NULL,
            filename     TEXT    NOT NULL,
            label        TEXT    NOT NULL,
            score        REAL    NOT NULL,
            route        TEXT    NOT NULL,
            arch         TEXT    NOT NULL,
            latency_ms   REAL    NOT NULL,
            thumbnail_png BLOB,
            heatmap_json  TEXT,
            batch_tag    TEXT
        )
    """)
    conn.commit()
    conn.close()


def insert_history(filename, result, batch_tag=None):
    """把一条推理结果写入数据库."""
    thumb = inference.make_thumbnail_png_bytes(result["preprocessed"])
    heat  = result["heatmap"]
    heat_json = json.dumps(heat.tolist()) if heat is not None else None

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO history
          (timestamp, filename, label, score, route, arch, latency_ms,
           thumbnail_png, heatmap_json, batch_tag)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        filename, result["label"], result["score"], result["route"],
        result["arch"], result["latency_ms"],
        thumb, heat_json, batch_tag,
    ))
    conn.commit()
    conn.close()


def query_history(limit=200, batch_tag=None):
    """返回最近 N 条历史 (DataFrame)."""
    conn = sqlite3.connect(DB_PATH)
    q = """SELECT id, timestamp, filename, label, score, route, arch,
                  latency_ms, batch_tag
           FROM history"""
    params = []
    if batch_tag is not None:
        q += " WHERE batch_tag = ?"
        params.append(batch_tag)
    q += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()
    return df


def load_history_record(record_id):
    """根据 id 拉一条完整记录 (含 thumbnail + heatmap)."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM history WHERE id = ?", (record_id,))
    row = cur.fetchone()
    cols = [c[0] for c in cur.description]
    conn.close()
    if row is None:
        return None
    return dict(zip(cols, row))


def clear_all_history():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM history")
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 模型加载 (Streamlit 缓存, 整个进程只加载一次)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="加载模型中…")
def get_detector(arch):
    """加载并缓存 detector. arch ∈ {'mil', 'simple'}."""
    return inference.GANDetector(arch=arch)


# ─────────────────────────────────────────────────────────────────────────────
# 公共 UI 工具
# ─────────────────────────────────────────────────────────────────────────────

def render_result_card(result, filename=None, show_heatmap=True):
    """渲染一条推理结果 (用于单张检测和历史详情)."""
    label = result["label"]
    fake_prob = result["fake_prob"]
    score = result["score"]
    route = result["route"]
    arch  = result["arch"]
    heat  = result["heatmap"]
    img128 = result["preprocessed"]

    # 顶部状态条
    if label == "fake":
        st.error(f"🚨 判定: **FAKE (AI 生成)**   "
                 f"P(fake) = {fake_prob:.4f}")
    else:
        st.success(f"✅ 判定: **REAL (真实)**   "
                   f"P(real) = {score:.4f}")

    # 元信息
    cols = st.columns(4)
    cols[0].metric("置信度", f"{max(fake_prob, score):.4f}")
    cols[1].metric("路由", route.upper(),
                   help="FFT = 频域快速判定; CNN = 神经网络精判")
    cols[2].metric("架构", arch.upper())
    cols[3].metric("延迟", f"{result['latency_ms']:.1f} ms")

    if not show_heatmap:
        st.image(img128, caption="预处理后图像 (128×128 灰度)",
                 width=300)
        return

    # 热力图 / 原图 / overlay 三选一
    if heat is None:
        st.info("ℹ️ 当前架构 (SimpleCNN) 不支持热力图。"
                "切换到 MIL 架构可查看 patch 级伪造证据图。")
        st.image(img128, caption="预处理后图像 (128×128 灰度)",
                 width=300)
    else:
        view_mode = st.radio(
            "视图模式",
            options=["overlay", "image", "heatmap"],
            format_func=lambda x: {"overlay": "🎨 叠加显示",
                                    "image":   "📷 原图",
                                    "heatmap": "🔥 仅热力图"}[x],
            horizontal=True,
            key=f"view_mode_{filename or id(result)}",
        )

        c1, c2 = st.columns([2, 1])
        with c1:
            rgb = inference.render_heatmap_overlay(
                img128, heat, mode=view_mode)
            st.image(rgb, caption=f"{view_mode}",
                     use_column_width=True)
        with c2:
            st.markdown("**4×4 原始 attention × P(real|patch) 数值表**")
            df_heat = pd.DataFrame(
                heat.astype(np.float32),
                columns=[f"col{i}" for i in range(4)],
                index=[f"row{i}" for i in range(4)],
            )
            st.dataframe(
                df_heat.style.background_gradient(cmap="jet", vmin=0, vmax=heat.max()).format("{:.4f}"),
                use_container_width=True,
            )
            st.caption(f"Heatmap 总和: {heat.sum():.4f}    "
                       f"最大: {heat.max():.4f}    "
                       f"最小: {heat.min():.4f}")
            st.caption(
                "💡 数值 = attention × P(real|patch). "
                "高 = 重要且像 real; 低 = 不重要 或 像 fake.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: 单张检测
# ─────────────────────────────────────────────────────────────────────────────

def page_single():
    st.header("📤 单张图片检测")
    st.write("上传一张 jpg/png 图片，模型会自动检测它是否为 GAN 生成。"
             "或者直接点下方按钮加载论文示例。")

    # ── 示例图片快捷加载 ─────────────────────────────────────────────────
    fake_paths = _list_sample_image_paths("fake", 2)
    real_paths = _list_sample_image_paths("real", 2)
    cols = st.columns(4)
    sample_buttons = [
        (cols[0], "🤖 示例 FAKE #1", fake_paths[0] if len(fake_paths) > 0 else None),
        (cols[1], "🤖 示例 FAKE #2", fake_paths[1] if len(fake_paths) > 1 else None),
        (cols[2], "👤 示例 REAL #1", real_paths[0] if len(real_paths) > 0 else None),
        (cols[3], "👤 示例 REAL #2", real_paths[1] if len(real_paths) > 1 else None),
    ]
    for col, label, path in sample_buttons:
        if col.button(label, use_container_width=True,
                      disabled=(path is None)):
            with open(path, "rb") as f:
                st.session_state.single_demo_bytes = f.read()
                st.session_state.single_demo_name  = os.path.basename(path)
            _safe_rerun()

    uploaded = st.file_uploader(
        "上传图片",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="single_uploader",
    )

    # 优先使用上传文件；否则使用会话里缓存的示例
    if uploaded is not None:
        st.session_state.pop("single_demo_bytes", None)
        st.session_state.pop("single_demo_name", None)
        if uploaded.size > SINGLE_MAX_BYTES:
            st.error(f"文件过大 ({uploaded.size/1024/1024:.1f} MB)，"
                     f"上限 {SINGLE_MAX_BYTES/1024/1024:.0f} MB。")
            return
        bytes_data = uploaded.read()
        filename   = uploaded.name
    elif "single_demo_bytes" in st.session_state:
        bytes_data = st.session_state.single_demo_bytes
        filename   = st.session_state.single_demo_name
        st.caption(f"📁 当前使用论文示例：`{filename}`")
    else:
        return

    # 显示原图（默认展开，便于截图）
    with st.expander("🖼️ 原始图片", expanded=True):
        st.image(bytes_data, caption=filename, width=300)

    # 推理
    with st.spinner("推理中…"):
        det = get_detector(st.session_state.get("arch", "mil"))
        try:
            result = det.predict_single(bytes_data)
        except Exception as e:
            st.error(f"推理失败: {type(e).__name__}: {e}")
            return

    # 写入数据库
    insert_history(filename, result, batch_tag=None)

    # 渲染结果
    st.divider()
    render_result_card(result, filename=filename)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: 批量 zip 检测
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_image_member(name):
    """zip 成员名是否合法且为图片. 防止路径穿越和非图片文件."""
    # 拒绝绝对路径和 ../
    if name.startswith("/") or name.startswith("\\"):
        return False
    if ".." in name.replace("\\", "/").split("/"):
        return False
    # 拒绝目录条目
    if name.endswith("/") or name.endswith("\\"):
        return False
    # 只接受白名单后缀
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        return False
    # 拒绝 macOS 隐藏文件
    base = os.path.basename(name)
    if base.startswith("."):
        return False
    if "__MACOSX" in name:
        return False
    return True


def extract_images_from_zip(zip_bytes):
    """
    返回 list of (filename, image_bytes), 已过滤非图片.
    """
    out = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if not is_valid_image_member(info.filename):
                continue
            try:
                data = zf.read(info)
                # 验证能 PIL 打开
                Image.open(io.BytesIO(data)).verify()
                out.append((info.filename, data))
            except Exception:
                continue
            if len(out) >= ZIP_MAX_IMAGES:
                break
    return out


def page_batch():
    st.header("📦 批量检测 (zip)")
    st.write(f"上传一个 zip 压缩包，最多处理 **{ZIP_MAX_IMAGES} 张**图片。"
             f"非 jpg/png/bmp/webp 文件会被自动忽略。"
             f"或者点右侧按钮加载论文示例 ZIP。")

    c_left, c_right = st.columns([3, 1])
    uploaded = c_left.file_uploader(
        "上传 zip 文件",
        type=["zip"],
        key="batch_uploader",
    )
    use_demo = c_right.button(
        "🚀 加载示例 ZIP\n(10 fake + 10 real)",
        use_container_width=True,
    )

    if uploaded is not None:
        zip_bytes = uploaded.read()
    elif use_demo:
        zip_bytes = _build_sample_zip(10, 10)
        st.success("已加载论文示例 ZIP（共 20 张）")
    else:
        return

    # 解压
    with st.spinner("解压并扫描图片…"):
        try:
            images = extract_images_from_zip(zip_bytes)
        except zipfile.BadZipFile:
            st.error("zip 文件损坏或格式错误。")
            return

    if len(images) == 0:
        st.warning("zip 中没有找到合法的图片文件。")
        return

    n = len(images)
    st.info(f"找到 **{n}** 张可处理图片")

    # 示例 ZIP 自动开始；用户上传的需点按钮确认
    auto_run = use_demo
    start = auto_run or st.button(
        f"开始批量检测 ({n} 张)", type="primary",
        use_container_width=True)

    if start:

        det = get_detector(st.session_state.get("arch", "mil"))
        prog = st.progress(0.0, text="推理中… 0/0")
        status = st.empty()

        # 用一个 batch_tag 把这次批量的所有结果聚合
        batch_tag = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        names    = [name for name, _ in images]
        bytes_list = [b   for _, b   in images]

        # 一次性 predict_batch (内部用 cascade)
        t0 = time.time()
        try:
            results = det.predict_batch(
                bytes_list,
                on_progress=lambda done, total: (
                    prog.progress(done / total,
                                  text=f"推理中… {done}/{total}"),
                    status.caption(f"正在处理: {names[done-1]}"),
                ),
            )
        except Exception as e:
            st.error(f"批量推理失败: {type(e).__name__}: {e}")
            return
        elapsed = time.time() - t0

        prog.empty()
        status.empty()

        # 写库
        with st.spinner("保存到历史记录…"):
            for name, r in zip(names, results):
                insert_history(name, r, batch_tag=batch_tag)

        # 汇总
        n_fake = sum(1 for r in results if r["label"] == "fake")
        n_real = n - n_fake
        n_cnn  = sum(1 for r in results if r["route"] == "cnn")

        st.success(f"✅ 完成 {n} 张  耗时 {elapsed:.2f}s  "
                   f"(平均 {elapsed*1000/n:.1f} ms/张)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总计",         n)
        c2.metric("判为 FAKE",    n_fake, delta=f"{100*n_fake/n:.1f}%")
        c3.metric("判为 REAL",    n_real, delta=f"{100*n_real/n:.1f}%")
        c4.metric("走 CNN 比例",  f"{100*n_cnn/n:.1f}%")

        # 分布可视化（截图友好）
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("**判定分布**")
            st.bar_chart(
                pd.DataFrame({"数量": [n_fake, n_real]},
                             index=["FAKE", "REAL"]),
                color="#C9437A", height=240,
            )
        with chart_col2:
            st.markdown("**路由分布**")
            n_fft = n - n_cnn
            st.bar_chart(
                pd.DataFrame({"数量": [n_fft, n_cnn]},
                             index=["FFT 直接判决", "MIL-CNN 精判"]),
                color="#3D62C7", height=240,
            )

        # 结果表格
        df = pd.DataFrame([
            {
                "filename": name,
                "label":    r["label"],
                "P(fake)":  round(r["fake_prob"], 4),
                "P(real)":  round(r["score"],     4),
                "route":    r["route"],
                "latency_ms": round(r["latency_ms"], 1),
            }
            for name, r in zip(names, results)
        ])
        st.dataframe(df, use_container_width=True, height=400)

        # CSV 下载
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 下载结果 CSV",
            data=csv,
            file_name=f"{batch_tag}_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: 历史记录
# ─────────────────────────────────────────────────────────────────────────────

def render_history_detail(record):
    """渲染一条历史记录 (从 SQLite 拿到的 dict)."""
    st.markdown(f"**文件名**: `{record['filename']}`")
    st.markdown(f"**时间**: {record['timestamp']}    "
                f"**架构**: {record['arch'].upper()}    "
                f"**路由**: {record['route'].upper()}")

    c1, c2 = st.columns([1, 2])

    # 缩略图
    with c1:
        if record["thumbnail_png"]:
            thumb = Image.open(io.BytesIO(record["thumbnail_png"]))
            st.image(thumb, caption="预处理缩略图", width=200)
        st.markdown(
            f"- **判定**: {record['label'].upper()}\n"
            f"- **score** (P(real)): {record['score']:.4f}\n"
            f"- **P(fake)**: {1-record['score']:.4f}\n"
            f"- **latency**: {record['latency_ms']:.1f} ms"
        )

    # 热力图
    with c2:
        if record["heatmap_json"]:
            heat = np.array(json.loads(record["heatmap_json"]),
                            dtype=np.float32)
            thumb = np.array(Image.open(io.BytesIO(record["thumbnail_png"])))
            view_mode = st.radio(
                "视图",
                options=["overlay", "image", "heatmap"],
                format_func=lambda x: {"overlay": "🎨 叠加",
                                        "image":   "📷 原图",
                                        "heatmap": "🔥 热力图"}[x],
                horizontal=True,
                key=f"hist_view_{record['id']}",
            )
            rgb = inference.render_heatmap_overlay(
                thumb, heat, mode=view_mode)
            st.image(rgb, use_column_width=True)
            df_heat = pd.DataFrame(heat,
                                   columns=[f"c{i}" for i in range(4)],
                                   index=  [f"r{i}" for i in range(4)])
            with st.expander("4×4 原始热力图数值"):
                st.dataframe(
                    df_heat.style.background_gradient(
                        cmap="jet", vmin=0, vmax=heat.max()).format("{:.4f}"),
                    use_container_width=True)
        else:
            st.info("此记录无热力图 (SimpleCNN 架构)")


def page_history():
    st.header("📜 历史记录")

    c1, c2 = st.columns([3, 1])
    limit = c1.slider("显示最近多少条", 10, 500, 50)
    if c2.button("🗑️ 清空全部历史", type="secondary"):
        if st.session_state.get("confirm_clear", False):
            clear_all_history()
            st.session_state.confirm_clear = False
            st.success("已清空。")
            _safe_rerun()
        else:
            st.session_state.confirm_clear = True
            st.warning("再点一次确认清空。")

    df = query_history(limit=limit)
    if df.empty:
        st.info("暂无历史记录。先去做几次单张或批量检测吧！")
        return

    # 格式化
    df_show = df.copy()
    df_show["P(fake)"] = (1 - df_show["score"]).round(4)
    df_show["score"]   = df_show["score"].round(4)
    df_show["latency_ms"] = df_show["latency_ms"].round(1)
    df_show = df_show[["id", "timestamp", "filename", "label",
                       "P(fake)", "score", "route", "arch",
                       "latency_ms", "batch_tag"]]

    st.dataframe(df_show, use_container_width=True, height=300)

    # 看详情（默认展开最新一条，便于截图）
    st.divider()
    st.subheader("查看某条详情")
    options = [f"#{r.id} — {r.filename} ({r.label})"
               for r in df.itertuples(index=False)]
    pick = st.selectbox(
        "选择记录",
        options=options,
        index=0 if options else None,
    )
    if pick:
        rid = int(pick.split("#")[1].split(" ")[0])
        record = load_history_record(rid)
        if record:
            render_history_detail(record)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4: 模型设置
# ─────────────────────────────────────────────────────────────────────────────

def page_settings():
    st.header("⚙️ 模型与设置")

    avail = inference.check_models_available()

    st.subheader("模型可用性")
    c1, c2, c3 = st.columns(3)
    c1.metric("FFT-LogReg",  "✅ 可用" if avail["fft"]        else "❌ 缺失")
    c2.metric("MIL CNN",     "✅ 可用" if avail["mil_cnn"]    else "❌ 缺失")
    c3.metric("Simple CNN",  "✅ 可用" if avail["simple_cnn"] else "❌ 缺失")

    with st.expander("模型路径"):
        st.json(avail["details"])

    st.subheader("当前架构")
    cur_arch = st.session_state.get("arch", "mil")
    archs_available = []
    if avail["mil_cnn"]:
        archs_available.append("mil")
    if avail["simple_cnn"]:
        archs_available.append("simple")

    if not archs_available:
        st.error("没有任何 CNN 模型可用！请先训练并放到 ./runs/ 目录。")
        return

    if cur_arch not in archs_available:
        cur_arch = archs_available[0]
        st.session_state.arch = cur_arch

    new_arch = st.radio(
        "推理架构",
        options=archs_available,
        index=archs_available.index(cur_arch),
        format_func=lambda a: {
            "mil":    "MIL — 注意力多实例学习 (推荐, 支持热力图)",
            "simple": "SimpleCNN — 整图分类 (无热力图)",
        }[a],
    )
    if new_arch != cur_arch:
        st.session_state.arch = new_arch
        # 清掉缓存的 detector, 强制重新加载新架构
        get_detector.clear()
        st.success(f"已切换到 {new_arch.upper()},下次推理会重新加载。")

    st.divider()
    st.subheader("Cascade 阈值 (来自训练时 tune)")
    st.markdown(
        f"- **low_thr**  = `{inference.DEFAULT_LOW_THR}` "
        f"(FFT score < 此值 → 直接判 fake, 不调 CNN)\n"
        f"- **high_thr** = `{inference.DEFAULT_HIGH_THR}` "
        f"(FFT score > 此值 → 直接判 real, 不调 CNN)\n"
        f"- 中间区间 [{inference.DEFAULT_LOW_THR}, "
        f"{inference.DEFAULT_HIGH_THR}] 的样本会调 CNN 精判。"
    )

    st.subheader("数据库")
    st.text(f"路径: {DB_PATH}")
    if os.path.exists(DB_PATH):
        size_kb = os.path.getsize(DB_PATH) / 1024
        st.text(f"大小: {size_kb:.1f} KB")
        df = query_history(limit=10**9)
        st.text(f"总记录数: {len(df)}")


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    init_db()

    # 默认架构
    if "arch" not in st.session_state:
        st.session_state.arch = "mil"

    st.title("🔍 GAN 图像检测器")
    st.caption(
        "基于 FFT 频域特征 + Patch 级 Multi-Instance Attention 的 cascade pipeline. "
        "训练于 StyleGAN-FFHQ vs FFHQ.")

    # 顶部状态条
    avail = inference.check_models_available()
    if not avail["fft"]:
        st.error(f"❌ FFT 模型未找到: {avail['details']['fft_pkl']}")
        st.stop()
    if not avail["any_cnn"]:
        st.error("❌ 没有任何 CNN 模型可用,请检查 ./runs/ 目录")
        st.stop()

    cur_arch = st.session_state.arch
    if cur_arch == "mil" and not avail["mil_cnn"]:
        st.session_state.arch = "simple"
        cur_arch = "simple"
    if cur_arch == "simple" and not avail["simple_cnn"]:
        st.session_state.arch = "mil"
        cur_arch = "mil"

    st.sidebar.markdown(f"### 当前架构: **{cur_arch.upper()}**")
    st.sidebar.caption(
        "在 ⚙️ 模型与设置 中切换架构\n\n"
        "**MIL** — 含热力图,精度高\n\n"
        "**SimpleCNN** — 仅分类,速度快")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📤 单张检测", "📦 批量检测", "📜 历史记录", "⚙️ 设置"])

    with tab1:
        page_single()
    with tab2:
        page_batch()
    with tab3:
        page_history()
    with tab4:
        page_settings()


if __name__ == "__main__":
    main()