"""
Build algorithm-block XML and replace existing code paragraphs in document.xml.
Per thesis rules: 算法 X-Y, font 五号 (sz=21), input/output + numbered steps.
"""
import re

DOC = 'D:/design1/GANDCTAnalysis/unpacked_skill/word/document.xml'

# ── Algorithm blocks ────────────────────────────────────────────────────────
ALGO_3_1 = {
    'title': '算法 3-1  FFT 特征提取核心步骤',
    'lines': [
        ('input',  '输入：单通道灰度图像 img ∈ ℝ^(128×128)'),
        ('output', '输出：功率谱密度矩阵 PSD ∈ ℝ^(128×128)'),
        ('step',   '1. F ← FFT2(img)                    // 二维离散傅里叶变换'),
        ('step',   '2. F ← fftshift(F)                  // 将零频分量移至频谱中心'),
        ('step',   '3. PSD ← (Re(F))² + (Im(F))²        // 计算功率谱密度'),
        ('step',   '4. return PSD'),
    ],
}

ALGO_A1_1 = {
    'title': '算法 A1-1  FFT/PSD/HFE 特征提取算法',
    'lines': [
        ('input',  '输入：图像批量 imgs ∈ ℝ^(N×H×W)，径向缓存 cache，HFE 截止比例 hfe_ratio'),
        ('output', '输出：特征矩阵 feats ∈ ℝ^(N×65)'),
        ('step',   '1.  F ← FFT2(imgs, axes=(-2, -1))'),
        ('step',   '2.  F ← fftshift(F, axes=(-2, -1))'),
        ('step',   '3.  PSD ← (Re(F))² + (Im(F))²                                  // 形状 (N, H, W)'),
        ('step',   '4.  prof ← radial_profile_batch(PSD, cache)                    // 径向 binning，输出 (N, 64)'),
        ('step',   '5.  prof ← log(1 + prof)                                       // 对数压缩动态范围'),
        ('step',   '6.  bins ← prof.shape[1];  k₀ ← ⌊bins × (1 - hfe_ratio)⌋'),
        ('step',   '7.  hfe ← Σ(prof[:, k₀:], axis=1) / (Σ(prof, axis=1) + ε)      // 高频能量比'),
        ('step',   '8.  μ ← mean(prof, axis=1);  σ ← std(prof, axis=1) + ε'),
        ('step',   '9.  prof ← (prof − μ) / σ                                      // z-score 归一化'),
        ('step',   '10. feats ← concat(prof, hfe.reshape(N, 1), axis=1)'),
        ('step',   '11. return feats'),
    ],
}

ALGO_A1_2 = {
    'title': '算法 A1-2  FFT→CNN 级联推理算法',
    'lines': [
        ('input',  '输入：图像批量 imgs ∈ ℝ^(N×H×W)，FFT 模型 fft_lr，MIL-CNN 模型 mil，路由阈值 (θ_low, θ_high)'),
        ('output', '输出：标签 labels，置信度 scores，热力图 heatmaps，路由 routes'),
        ('step',   '1.  feats ← 算法 A1-1(imgs)                                      // FFT/PSD/HFE 特征提取'),
        ('step',   '2.  fft_scores ← fft_lr.predict_proba(feats)[:, 1]              // 预测 P(real) ∈ [0, 1]'),
        ('step',   '3.  ambiguous ← (fft_scores ≥ θ_low) ∧ (fft_scores ≤ θ_high)    // 模糊样本掩码'),
        ('step',   '4.  routes ← where(ambiguous, "cnn", "fft")'),
        ('step',   '5.  scores ← copy(fft_scores);  heatmaps ← [None] × N'),
        ('step',   '6.  if any(ambiguous) then'),
        ('step',   '7.      amb_imgs ← imgs[ambiguous]'),
        ('step',   '8.      cnn_scores, cnn_heats ← mil.forward(amb_imgs)           // MIL-CNN 精判'),
        ('step',   '9.      for j, idx in enumerate(indices_where(ambiguous)) do'),
        ('step',   '10.         scores[idx] ← cnn_scores[j]'),
        ('step',   '11.         heatmaps[idx] ← cnn_heats[j]'),
        ('step',   '12.     end for'),
        ('step',   '13. end if'),
        ('step',   '14. labels ← where(scores ≥ 0.5, "real", "fake")'),
        ('step',   '15. return labels, scores, heatmaps, routes'),
    ],
}


def build_algorithm_xml(algo):
    """Generate the XML for one algorithm block."""
    parts = []
    # Title — top + bottom border, centered, bold, 五号
    parts.append(f'''    <w:p>
      <w:pPr>
        <w:pBdr>
          <w:top w:val="single" w:sz="6" w:space="6" w:color="auto"/>
          <w:bottom w:val="single" w:sz="6" w:space="6" w:color="auto"/>
        </w:pBdr>
        <w:spacing w:before="120" w:after="60" w:line="320" w:lineRule="exact"/>
        <w:jc w:val="center"/>
      </w:pPr>
      <w:r>
        <w:rPr>
          <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/>
          <w:b/><w:bCs/>
          <w:sz w:val="21"/>
        </w:rPr>
        <w:t xml:space="preserve">{algo["title"]}</w:t>
      </w:r>
    </w:p>''')

    # Body lines
    n = len(algo['lines'])
    for i, (kind, text) in enumerate(algo['lines']):
        is_last = (i == n - 1)
        bdr = ''
        if is_last:
            bdr = '''<w:pBdr>
          <w:bottom w:val="single" w:sz="6" w:space="6" w:color="auto"/>
        </w:pBdr>
        '''
        # Use Courier-ish look for steps: keep monospace alignment via Consolas; keep input/output normal
        if kind == 'step':
            font_ascii = 'Consolas'
            ind = '<w:ind w:left="400" w:hanging="400"/>'
        else:
            font_ascii = 'Times New Roman'
            ind = '<w:ind w:firstLine="0"/>'

        parts.append(f'''    <w:p>
      <w:pPr>
        {bdr}<w:spacing w:before="0" w:after="0" w:line="320" w:lineRule="exact"/>
        {ind}
        <w:jc w:val="left"/>
      </w:pPr>
      <w:r>
        <w:rPr>
          <w:rFonts w:ascii="{font_ascii}" w:hAnsi="{font_ascii}" w:eastAsia="宋体"/>
          <w:sz w:val="21"/>
        </w:rPr>
        <w:t xml:space="preserve">{text}</w:t>
      </w:r>
    </w:p>''')
    return '\n'.join(parts)


def replace_paragraph_range(xml, start_marker_text, n_paragraphs, replacement_xml):
    """Find first paragraph containing start_marker_text, then remove that paragraph
    and the next (n_paragraphs-1) paragraphs, replacing with replacement_xml."""
    idx = xml.find(start_marker_text)
    if idx == -1:
        raise ValueError(f'Marker not found: {start_marker_text!r}')
    # Find paragraph start
    p_start = xml.rfind('<w:p ', 0, idx)
    p_alt = xml.rfind('<w:p>', 0, idx)
    if p_alt > p_start:
        p_start = p_alt
    # Find end of n_paragraphs paragraphs from p_start
    cur = p_start
    for _ in range(n_paragraphs):
        cur = xml.find('</w:p>', cur) + len('</w:p>')
    p_end = cur
    return xml[:p_start] + replacement_xml + xml[p_end:]


def main():
    with open(DOC, encoding='utf-8') as f:
        xml = f.read()

    # Replace section 3.3.1 inline FFT code (3 paragraphs)
    xml = replace_paragraph_range(xml, 'F = np.fft.fft2(img)', 3, build_algorithm_xml(ALGO_3_1))
    print('replaced 3.3.1 inline code with 算法 3-1')

    # Replace appendix A1.1 code (single big paragraph)
    xml = replace_paragraph_range(xml, 'def build_radial_cache', 1, build_algorithm_xml(ALGO_A1_1))
    print('replaced A1.1 code with 算法 A1-1')

    # Replace appendix A1.2 code (single big paragraph)
    xml = replace_paragraph_range(xml, '# 步骤2: FFT 特征提取', 1, build_algorithm_xml(ALGO_A1_2))
    print('replaced A1.2 code with 算法 A1-2')

    with open(DOC, 'w', encoding='utf-8') as f:
        f.write(xml)
    print('saved.')


if __name__ == '__main__':
    main()
