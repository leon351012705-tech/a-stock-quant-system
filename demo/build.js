/* A股量化交易系统 — 求职作品集 PPT 生成脚本 */
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const Fa = require("react-icons/fa");

// ── 调色板：Midnight Quant（深海军蓝 + 琥珀强调）──
const NAVY_DARK = "11203F";   // 深色页背景
const NAVY      = "1F2D4E";   // 主标题/深色块
const NAVY_MID  = "2E4372";   // 次级海军蓝
const STEEL     = "4C6699";   // 钢蓝
const ICE       = "B9CCEA";   // 浅蓝（深色背景上的文字）
const AMBER     = "E8A33D";   // 琥珀强调色
const WHITE     = "FFFFFF";
const CARD      = "F2F5FA";   // 浅色卡片
const BORDER    = "E0E5EF";   // 卡片描边
const MUTED     = "707B8F";   // 弱化文字
const INK       = "232F47";   // 正文深色

const HF = "Trebuchet MS";    // 标题字体
const BF = "Calibri";         // 正文字体
const MONO = "Consolas";      // 数据/代码字体

// ── 图标转 base64 PNG ──
async function icon(IconComponent, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
  const png = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + png.toString("base64");
}

const makeShadow = () => ({ type: "outer", color: "1A2540", blur: 7, offset: 3, angle: 135, opacity: 0.16 });

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";       // 10 x 5.625
  pres.author = "Quant System Portfolio";
  pres.title  = "A股量化交易系统 — 作品集";

  // 预渲染图标
  const ic = {
    db:       await icon(Fa.FaDatabase, "#" + WHITE),
    chart:    await icon(Fa.FaChartLine, "#" + WHITE),
    layers:   await icon(Fa.FaLayerGroup, "#" + WHITE),
    filter:   await icon(Fa.FaFilter, "#" + WHITE),
    bell:     await icon(Fa.FaBell, "#" + WHITE),
    feed:     await icon(Fa.FaStream, "#" + WHITE),
    bolt:     await icon(Fa.FaBolt, "#" + AMBER),
    shield:   await icon(Fa.FaShieldAlt, "#" + AMBER),
    flask:    await icon(Fa.FaFlask, "#" + AMBER),
    code:     await icon(Fa.FaCode, "#" + AMBER),
    cogs:     await icon(Fa.FaCogs, "#" + AMBER),
    server:   await icon(Fa.FaServer, "#" + AMBER),
    sync:     await icon(Fa.FaSyncAlt, "#" + AMBER),
    calendar: await icon(Fa.FaRegCalendarCheck, "#" + AMBER),
    mail:     await icon(Fa.FaEnvelopeOpenText, "#" + AMBER),
    check:    await icon(Fa.FaCheckCircle, "#" + AMBER),
    py:       await icon(Fa.FaPython, "#" + AMBER),
  };

  // ════════════════════════════════════════════════════
  //  通用：内容页页眉
  // ════════════════════════════════════════════════════
  function header(slide, kicker, title) {
    slide.addShape(pres.shapes.RECTANGLE, { x: 0.6, y: 0.46, w: 0.16, h: 0.16, fill: { color: AMBER } });
    slide.addText(kicker.toUpperCase(), {
      x: 0.86, y: 0.40, w: 8.5, h: 0.28, margin: 0,
      fontFace: BF, fontSize: 11, bold: true, color: AMBER, charSpacing: 3,
    });
    slide.addText(title, {
      x: 0.84, y: 0.62, w: 8.7, h: 0.62, margin: 0,
      fontFace: HF, fontSize: 28, bold: true, color: NAVY,
    });
  }
  function footer(slide, n) {
    slide.addText("A股量化交易系统", {
      x: 0.6, y: 5.28, w: 4, h: 0.3, margin: 0,
      fontFace: BF, fontSize: 9, color: MUTED,
    });
    slide.addText(String(n).padStart(2, "0") + " / 12", {
      x: 8.4, y: 5.28, w: 1.0, h: 0.3, margin: 0,
      fontFace: MONO, fontSize: 9, color: MUTED, align: "right",
    });
  }

  // ════════════════════════════════════════════════════
  //  Slide 1 — 封面
  // ════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: NAVY_DARK };

    // 右上角装饰方块
    for (let i = 0; i < 5; i++) {
      s.addShape(pres.shapes.RECTANGLE, {
        x: 8.55 + (i % 3) * 0.34, y: 0.5 + Math.floor(i / 3) * 0.34,
        w: 0.2, h: 0.2, fill: { color: AMBER, transparency: i * 14 },
      });
    }

    s.addText("个人作品集  ·  PORTFOLIO", {
      x: 0.75, y: 0.78, w: 6, h: 0.3, margin: 0,
      fontFace: BF, fontSize: 12, bold: true, color: AMBER, charSpacing: 3,
    });

    s.addText("A 股量化交易系统", {
      x: 0.72, y: 1.55, w: 9, h: 1.0, margin: 0,
      fontFace: HF, fontSize: 52, bold: true, color: WHITE,
    });
    s.addText("从数据采集到信号推送的全自动决策引擎", {
      x: 0.75, y: 2.62, w: 9, h: 0.5, margin: 0,
      fontFace: BF, fontSize: 19, color: ICE,
    });

    // 关键数字条
    const stats = [
      ["4000+", "行 Python 代码"],
      ["5520", "只 A 股全覆盖"],
      ["728 万", "行历史日线数据"],
      ["6 层", "模块化架构"],
    ];
    const cw = 2.06, gap = 0.18, x0 = 0.75;
    stats.forEach(([num, label], i) => {
      const x = x0 + i * (cw + gap);
      s.addShape(pres.shapes.RECTANGLE, {
        x, y: 3.62, w: cw, h: 1.16, fill: { color: NAVY },
        line: { color: NAVY_MID, width: 1 },
      });
      s.addShape(pres.shapes.RECTANGLE, { x, y: 3.62, w: 0.07, h: 1.16, fill: { color: AMBER } });
      s.addText(num, {
        x: x + 0.2, y: 3.74, w: cw - 0.3, h: 0.5, margin: 0,
        fontFace: HF, fontSize: 27, bold: true, color: AMBER,
      });
      s.addText(label, {
        x: x + 0.2, y: 4.24, w: cw - 0.3, h: 0.4, margin: 0,
        fontFace: BF, fontSize: 12, color: ICE,
      });
    });

    s.addText("独立设计 · 独立开发 · 独立维护   |   Python · SQLite · 多源行情数据", {
      x: 0.75, y: 5.0, w: 9, h: 0.3, margin: 0,
      fontFace: BF, fontSize: 11, color: STEEL,
    });
  }

  // ════════════════════════════════════════════════════
  //  Slide 2 — 项目概览
  // ════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    header(s, "Overview", "项目概览");

    // 左侧：定位 + 要点
    s.addText("一个人从零设计、开发、维护的 A 股量化决策系统。每日盘后自动拉取全市场行情，运行多策略扫描，输出经过多维评分的交易信号。", {
      x: 0.86, y: 1.42, w: 4.95, h: 1.0, margin: 0,
      fontFace: BF, fontSize: 13.5, color: INK, lineSpacingMultiple: 1.25,
    });

    const points = [
      ["定位", "半自动决策助手 — 系统选股，人工拍板"],
      ["运行方式", "每日 17:30 一键运行，~80 秒完成全流程"],
      ["数据规模", "5520 只股票 × 6 年日线 = 728 万行"],
      ["交付产物", "每日信号邮件，含星级评分与风险标签"],
    ];
    let py = 2.55;
    points.forEach(([k, v]) => {
      s.addShape(pres.shapes.RECTANGLE, { x: 0.86, y: py + 0.04, w: 0.13, h: 0.13, fill: { color: AMBER } });
      s.addText(k, {
        x: 1.12, y: py - 0.07, w: 1.5, h: 0.34, margin: 0,
        fontFace: BF, fontSize: 12.5, bold: true, color: NAVY,
      });
      s.addText(v, {
        x: 2.35, y: py - 0.07, w: 3.5, h: 0.34, margin: 0,
        fontFace: BF, fontSize: 12, color: INK,
      });
      py += 0.62;
    });

    // 右侧：2x2 数字卡
    const cards = [
      ["28", "Python 模块", ic.code],
      ["4", "交易策略并行", ic.cogs],
      ["3", "数据源容错切换", ic.server],
      ["100%", "全流程自动化", ic.sync],
    ];
    const CW = 2.18, CH = 1.62, GX = 0.22, GY = 0.22, X0 = 6.05, Y0 = 1.42;
    cards.forEach(([num, label, iconData], i) => {
      const x = X0 + (i % 2) * (CW + GX);
      const y = Y0 + Math.floor(i / 2) * (CH + GY);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x, y, w: CW, h: CH, rectRadius: 0.08,
        fill: { color: CARD }, line: { color: BORDER, width: 1 },
        shadow: makeShadow(),
      });
      s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.2, w: 0.5, h: 0.5, fill: { color: NAVY } });
      s.addImage({ data: iconData, x: x + 0.31, y: y + 0.31, w: 0.28, h: 0.28 });
      s.addText(num, {
        x: x + 0.18, y: y + 0.72, w: CW - 0.36, h: 0.5, margin: 0,
        fontFace: HF, fontSize: 26, bold: true, color: NAVY,
      });
      s.addText(label, {
        x: x + 0.2, y: y + 1.18, w: CW - 0.36, h: 0.32, margin: 0,
        fontFace: BF, fontSize: 11.5, color: MUTED,
      });
    });

    footer(s, 2);
  }

  // ════════════════════════════════════════════════════
  //  Slide 3 — 六层系统架构
  // ════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    header(s, "Architecture", "六层系统架构");

    s.addText("数据自下而上流动 — 每一层只依赖下层接口，职责单一、可独立替换", {
      x: 0.86, y: 1.24, w: 8.6, h: 0.3, margin: 0,
      fontFace: BF, fontSize: 12, color: MUTED,
    });

    const layers = [
      ["06", "输出层", "邮件 / 微信推送 — 美化排版，星级评分排序", ic.bell],
      ["05", "环境过滤层", "市场情绪两层判断 — 弱市自动停止推送", ic.filter],
      ["04", "共振层", "3 日窗口双确认 — boll_rv + 趋势策略叠加", ic.layers],
      ["03", "策略层", "4 策略并行 — MACD / 布林 / 均线 / SSB", ic.chart],
      ["02", "行情层", "增量更新 — 自动补 gap，节假日感知", ic.feed],
      ["01", "数据层", "SQLite 存储 — 728 万行，WAL + 索引优化", ic.db],
    ];
    const rowH = 0.52, gap = 0.065, x = 0.86, y0 = 1.60;
    layers.forEach(([no, name, desc, iconData], i) => {
      const y = y0 + i * (rowH + gap);
      // 渐变感：上层钢蓝，下层深海军
      const t = i / (layers.length - 1);
      const fill = i <= 1 ? STEEL : (i <= 3 ? NAVY_MID : NAVY);
      s.addShape(pres.shapes.RECTANGLE, { x, y, w: 8.28, h: rowH, fill: { color: fill } });
      // 序号块
      s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.62, h: rowH, fill: { color: AMBER } });
      s.addText(no, {
        x, y, w: 0.62, h: rowH, margin: 0, align: "center", valign: "middle",
        fontFace: HF, fontSize: 16, bold: true, color: NAVY_DARK,
      });
      // 图标
      s.addImage({ data: iconData, x: x + 0.84, y: y + 0.155, w: 0.25, h: 0.25 });
      // 名称
      s.addText(name, {
        x: x + 1.26, y, w: 1.85, h: rowH, margin: 0, valign: "middle",
        fontFace: HF, fontSize: 14.5, bold: true, color: WHITE,
      });
      // 描述
      s.addText(desc, {
        x: x + 3.1, y, w: 5.0, h: rowH, margin: 0, valign: "middle",
        fontFace: BF, fontSize: 11.5, color: ICE,
      });
    });

    footer(s, 3);
  }

  // ════════════════════════════════════════════════════
  //  Slide 4 — 技术栈
  // ════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    header(s, "Tech Stack", "技术栈选型");

    const groups = [
      ["语言与运行", ic.py, ["Python 3.12", "pandas / numpy 向量化", "Windows 计划任务调度"]],
      ["数据存储", ic.server, ["SQLite — WAL 模式", "idx 索引加速广度查询", "executemany 批量写入"]],
      ["行情数据源", ic.feed, ["Tushare — 主通道（全市场日 K）", "BaoStock — 备用通道", "akshare — 股票池 / 日历"]],
      ["信号与推送", ic.mail, ["SMTP 邮件推送（自带美化）", "Server酱 微信推送", "Excel 月度统计导出"]],
    ];
    const CW = 4.32, CH = 1.66, GX = 0.22, GY = 0.2, X0 = 0.86, Y0 = 1.4;
    groups.forEach(([title, iconData, items], i) => {
      const x = X0 + (i % 2) * (CW + GX);
      const y = Y0 + Math.floor(i / 2) * (CH + GY);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x, y, w: CW, h: CH, rectRadius: 0.07,
        fill: { color: CARD }, line: { color: BORDER, width: 1 }, shadow: makeShadow(),
      });
      s.addShape(pres.shapes.OVAL, { x: x + 0.26, y: y + 0.26, w: 0.56, h: 0.56, fill: { color: NAVY } });
      s.addImage({ data: iconData, x: x + 0.39, y: y + 0.39, w: 0.3, h: 0.3 });
      s.addText(title, {
        x: x + 1.0, y: y + 0.3, w: CW - 1.2, h: 0.5, margin: 0, valign: "middle",
        fontFace: HF, fontSize: 16, bold: true, color: NAVY,
      });
      s.addText(
        items.map((t, j) => ({ text: t, options: { bullet: { code: "2022" }, breakLine: j < items.length - 1, color: INK } })),
        { x: x + 1.0, y: y + 0.84, w: CW - 1.25, h: 0.72, margin: 0,
          fontFace: BF, fontSize: 11, color: INK, paraSpaceAfter: 4 }
      );
    });

    footer(s, 4);
  }

  // ════════════════════════════════════════════════════
  //  Slide 5 — 数据层工程
  // ════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    header(s, "Data Layer", "工程化的数据底座");

    // 左：三个大数字
    const big = [
      ["728 万", "行日线数据，覆盖 2020 年至今"],
      ["< 1 ms", "单日全市场广度查询（优化前 100-500ms）"],
      ["1.3 GB", "SQLite 单库，WAL 模式读写不阻塞"],
    ];
    let by = 1.5;
    big.forEach(([num, label]) => {
      s.addText(num, {
        x: 0.86, y: by, w: 2.6, h: 0.62, margin: 0,
        fontFace: HF, fontSize: 30, bold: true, color: AMBER,
      });
      s.addText(label, {
        x: 0.9, y: by + 0.6, w: 4.4, h: 0.5, margin: 0,
        fontFace: BF, fontSize: 11.5, color: INK, lineSpacingMultiple: 1.15,
      });
      by += 1.18;
    });

    // 右：三数据源容错链
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 5.55, y: 1.42, w: 3.92, h: 3.46, rectRadius: 0.08,
      fill: { color: NAVY }, line: { color: NAVY_MID, width: 1 }, shadow: makeShadow(),
    });
    s.addText("三层数据源容错链", {
      x: 5.8, y: 1.62, w: 3.5, h: 0.4, margin: 0,
      fontFace: HF, fontSize: 15, bold: true, color: WHITE,
    });
    const chain = [
      ["Tushare", "主通道 · 1 次 API 取全市场", AMBER],
      ["BaoStock", "备用 · 探测前哨防空跑", STEEL],
      ["精确诊断", "区分『未发布 / 节假日 / 异常』", ICE],
    ];
    let cy = 2.18;
    chain.forEach(([name, desc, col], i) => {
      s.addShape(pres.shapes.RECTANGLE, { x: 5.82, y: cy, w: 0.1, h: 0.66, fill: { color: col } });
      s.addText(name, {
        x: 6.04, y: cy - 0.02, w: 3.3, h: 0.32, margin: 0,
        fontFace: MONO, fontSize: 13, bold: true, color: WHITE,
      });
      s.addText(desc, {
        x: 6.04, y: cy + 0.28, w: 3.3, h: 0.34, margin: 0,
        fontFace: BF, fontSize: 10.5, color: ICE,
      });
      if (i < chain.length - 1) {
        s.addText("▼", { x: 5.74, y: cy + 0.62, w: 0.3, h: 0.28, margin: 0,
          fontFace: BF, fontSize: 10, color: STEEL, align: "center" });
      }
      cy += 0.92;
    });

    footer(s, 5);
  }

  // ════════════════════════════════════════════════════
  //  Slide 6 — 策略层 + 信号层
  // ════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    header(s, "Strategy & Signal", "策略层 + 信号层");

    // 4 策略卡（2x2）
    const strat = [
      ["MACD 金叉", "趋势", "DIFF 上穿 DEA，零轴下方更可靠"],
      ["布林带超跌反弹", "反转", "跌破下轨后收回，超跌反弹确认"],
      ["均线多头排列", "趋势", "MA5>10>20>60，趋势刚启动入场"],
      ["SSB 趋势回踩", "独立", "上升趋势中回踩 MA20，缩量企稳"],
    ];
    const CW = 2.92, CH = 1.16, GX = 0.16, GY = 0.16, X0 = 0.86, Y0 = 1.4;
    strat.forEach(([name, tag, desc], i) => {
      const x = X0 + (i % 2) * (CW + GX);
      const y = Y0 + Math.floor(i / 2) * (CH + GY);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x, y, w: CW, h: CH, rectRadius: 0.07,
        fill: { color: CARD }, line: { color: BORDER, width: 1 },
      });
      s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.07, h: CH, fill: { color: AMBER } });
      s.addText(name, {
        x: x + 0.22, y: y + 0.13, w: 1.95, h: 0.36, margin: 0,
        fontFace: HF, fontSize: 13.5, bold: true, color: NAVY,
      });
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: x + CW - 0.78, y: y + 0.16, w: 0.62, h: 0.28, rectRadius: 0.04,
        fill: { color: NAVY_MID },
      });
      s.addText(tag, {
        x: x + CW - 0.78, y: y + 0.16, w: 0.62, h: 0.28, margin: 0,
        align: "center", valign: "middle", fontFace: BF, fontSize: 9, bold: true, color: WHITE,
      });
      s.addText(desc, {
        x: x + 0.22, y: y + 0.52, w: CW - 0.4, h: 0.55, margin: 0,
        fontFace: BF, fontSize: 10.5, color: INK, lineSpacingMultiple: 1.15,
      });
    });

    // 右侧：共振 + 多维评分
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 6.98, y: 1.4, w: 2.5, h: 2.48, rectRadius: 0.08,
      fill: { color: NAVY }, shadow: makeShadow(),
    });
    s.addText("信号层核心", {
      x: 7.2, y: 1.56, w: 2.1, h: 0.34, margin: 0,
      fontFace: HF, fontSize: 13, bold: true, color: AMBER,
    });
    s.addText([
      { text: "共振规则 V4", options: { bold: true, color: WHITE, breakLine: true, fontSize: 11.5 } },
      { text: "boll_rv 必须命中 + 至少一个趋势策略", options: { color: ICE, breakLine: true, fontSize: 10 } },
      { text: " ", options: { breakLine: true, fontSize: 6 } },
      { text: "多维评分 Ranker", options: { bold: true, color: WHITE, breakLine: true, fontSize: 11.5 } },
      { text: "新鲜度 / 共振强度 / 趋势 / 流动性 / 风险 五维加权", options: { color: ICE, fontSize: 10 } },
    ], { x: 7.2, y: 1.96, w: 2.1, h: 1.8, margin: 0, lineSpacingMultiple: 1.2 });

    // 底部条
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.86, y: 4.08, w: 8.62, h: 0.78, rectRadius: 0.06,
      fill: { color: CARD }, line: { color: BORDER, width: 1 },
    });
    s.addImage({ data: ic.check, x: 1.04, y: 4.27, w: 0.4, h: 0.4 });
    s.addText([
      { text: "板块感知涨跌停  ", options: { bold: true, color: NAVY } },
      { text: "主板 10% / 创业·科创 20% / ST 5% — 按板块自动判定，避免误过滤强势股", options: { color: INK } },
    ], { x: 1.58, y: 4.08, w: 7.8, h: 0.78, margin: 0, valign: "middle", fontFace: BF, fontSize: 11 });

    footer(s, 6);
  }

  // ════════════════════════════════════════════════════
  //  Slide 7 — 工程亮点①：性能优化
  // ════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    header(s, "Highlight 01", "工程亮点 ① — 性能优化");

    // 两个大对比 callout
    const comp = [
      ["单日广度查询", "100 - 500 ms", "< 1 ms", "100x+"],
      ["节假日补数据", "45 分钟（卡死）", "15 秒", "180x"],
    ];
    comp.forEach(([title, before, after, mult], i) => {
      const x = 0.86 + i * 4.4;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x, y: 1.42, w: 4.18, h: 1.66, rectRadius: 0.08,
        fill: { color: NAVY }, shadow: makeShadow(),
      });
      s.addText(title, {
        x: x + 0.26, y: 1.56, w: 2.6, h: 0.34, margin: 0,
        fontFace: BF, fontSize: 12, bold: true, color: ICE,
      });
      s.addText(mult, {
        x: x + 2.7, y: 1.5, w: 1.32, h: 0.5, margin: 0, align: "right",
        fontFace: HF, fontSize: 25, bold: true, color: AMBER,
      });
      s.addText([
        { text: before + "  ", options: { color: STEEL, strike: true } },
        { text: "→  ", options: { color: MUTED } },
        { text: after, options: { color: WHITE, bold: true } },
      ], { x: x + 0.26, y: 2.1, w: 3.7, h: 0.6, margin: 0, fontFace: MONO, fontSize: 14, valign: "middle" });
    });

    // 优化手段
    s.addText("关键优化手段", {
      x: 0.86, y: 3.34, w: 4, h: 0.34, margin: 0,
      fontFace: HF, fontSize: 14, bold: true, color: NAVY,
    });
    const opt = [
      ["批量查询替代循环单查", "5499 次单股 SELECT → 1 次全量加载 + 内存 groupby"],
      ["SQLite WAL 模式 + 索引", "读写不阻塞；trade_date 索引加速广度查询"],
      ["executemany 批量写入", "逐行 INSERT → 单事务批量提交，写入快 5-10 倍"],
      ["BaoStock 探测前哨", "先探 2 只大盘股，节假日空跑 15 分钟 → 2 秒"],
    ];
    const OW = 4.32, OH = 0.66, OGX = 0.22, OGY = 0.14, OX0 = 0.86, OY0 = 3.74;
    opt.forEach(([k, v], i) => {
      const x = OX0 + (i % 2) * (OW + OGX);
      const y = OY0 + Math.floor(i / 2) * (OH + OGY);
      s.addShape(pres.shapes.RECTANGLE, { x, y, w: OW, h: OH, fill: { color: CARD }, line: { color: BORDER, width: 1 } });
      s.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.06, h: OH, fill: { color: AMBER } });
      s.addText(k, {
        x: x + 0.2, y: y + 0.07, w: OW - 0.3, h: 0.26, margin: 0,
        fontFace: BF, fontSize: 11, bold: true, color: NAVY,
      });
      s.addText(v, {
        x: x + 0.2, y: y + 0.32, w: OW - 0.3, h: 0.3, margin: 0,
        fontFace: BF, fontSize: 9.3, color: MUTED,
      });
    });

    footer(s, 7);
  }

  // ════════════════════════════════════════════════════
  //  Slide 8 — 工程亮点②：健壮性设计
  // ════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    header(s, "Highlight 02", "工程亮点 ② — 健壮性设计");

    s.addText("系统跑通只是起点 — 真正的工程量在于处理各种『非正常路径』", {
      x: 0.86, y: 1.24, w: 8.6, h: 0.3, margin: 0,
      fontFace: BF, fontSize: 12, color: MUTED,
    });

    const feats = [
      [ic.sync, "自动补 gap", "落后多个交易日时循环补齐，最多 30 轮防失控"],
      [ic.calendar, "节假日感知", "trade_calendar 表识别法定节假日，自动跳过"],
      [ic.shield, "数据完整性阈值", "动态按股票池 90% 校验，防部分故障被误判完整"],
      [ic.server, "三层数据源容错", "Tushare → BaoStock → 精确诊断，层层兜底"],
      [ic.bolt, "探测前哨", "全市场查询前先探测，避免无效空跑"],
      [ic.check, "时效性自愈", "每周一自动刷新股票名，捕捉 ST 戴帽/摘帽"],
    ];
    const CW = 2.78, CH = 1.32, GX = 0.14, GY = 0.16, X0 = 0.86, Y0 = 1.66;
    feats.forEach(([iconData, title, desc], i) => {
      const x = X0 + (i % 3) * (CW + GX);
      const y = Y0 + Math.floor(i / 3) * (CH + GY);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x, y, w: CW, h: CH, rectRadius: 0.07,
        fill: { color: CARD }, line: { color: BORDER, width: 1 }, shadow: makeShadow(),
      });
      s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.2, w: 0.46, h: 0.46, fill: { color: NAVY } });
      s.addImage({ data: iconData, x: x + 0.305, y: y + 0.305, w: 0.25, h: 0.25 });
      s.addText(title, {
        x: x + 0.78, y: y + 0.22, w: CW - 0.9, h: 0.42, margin: 0, valign: "middle",
        fontFace: HF, fontSize: 13, bold: true, color: NAVY,
      });
      s.addText(desc, {
        x: x + 0.22, y: y + 0.72, w: CW - 0.42, h: 0.5, margin: 0,
        fontFace: BF, fontSize: 10, color: INK, lineSpacingMultiple: 1.15,
      });
    });

    footer(s, 8);
  }

  // ════════════════════════════════════════════════════
  //  Slide 9 — 工程亮点③：数据驱动决策
  // ════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    header(s, "Highlight 03", "工程亮点 ③ — 数据驱动决策");

    s.addText("案例：『僵尸股过滤层』A/B 回测 — 用数据验证假设，而非凭直觉", {
      x: 0.86, y: 1.24, w: 8.6, h: 0.3, margin: 0,
      fontFace: BF, fontSize: 12, color: MUTED,
    });

    // A/B 对比图表
    s.addChart(pres.charts.BAR, [
      { name: "A 组 不过滤", labels: ["胜率 %", "累计收益 %"], values: [53.2, 56.8] },
      { name: "B 组 加过滤", labels: ["胜率 %", "累计收益 %"], values: [58.1, 29.5] },
    ], {
      x: 0.86, y: 1.66, w: 4.5, h: 2.7, barDir: "col",
      chartColors: [STEEL, AMBER],
      chartArea: { fill: { color: WHITE } },
      catAxisLabelColor: MUTED, catAxisLabelFontSize: 10, catAxisLabelFontFace: BF,
      valAxisLabelColor: MUTED, valAxisLabelFontSize: 9,
      valGridLine: { color: BORDER, size: 0.5 }, catGridLine: { style: "none" },
      showValue: true, dataLabelColor: NAVY, dataLabelFontSize: 9, dataLabelFontFace: BF,
      dataLabelPosition: "outEnd",
      showLegend: true, legendPos: "b", legendColor: INK, legendFontSize: 9,
      valAxisMaxVal: 70, valAxisMinVal: 0,
    });

    // 结论卡
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 5.6, y: 1.66, w: 3.88, h: 2.7, rectRadius: 0.08,
      fill: { color: NAVY }, shadow: makeShadow(),
    });
    s.addImage({ data: ic.flask, x: 5.84, y: 1.9, w: 0.36, h: 0.36 });
    s.addText("实验结论", {
      x: 6.32, y: 1.88, w: 3, h: 0.4, margin: 0, valign: "middle",
      fontFace: HF, fontSize: 15, bold: true, color: AMBER,
    });
    s.addText([
      { text: "过滤虽让胜率 +4.9%，但累计收益从 56.8% 砍到 29.5% — 几乎腰斩。", options: { color: WHITE, breakLine: true } },
      { text: " ", options: { breakLine: true, fontSize: 7 } },
      { text: "根因：用『过去 60 天没波动』判定僵尸，却误杀了『沉睡后觉醒』的慢牛。", options: { color: ICE, breakLine: true } },
      { text: " ", options: { breakLine: true, fontSize: 7 } },
      { text: "决策：尊重数据，不上线该模块。", options: { color: AMBER, bold: true } },
    ], { x: 5.84, y: 2.4, w: 3.42, h: 1.85, margin: 0, fontFace: BF, fontSize: 10.8, lineSpacingMultiple: 1.2 });

    // 底部方法论条
    s.addShape(pres.shapes.RECTANGLE, { x: 0.86, y: 4.56, w: 8.62, h: 0.5, fill: { color: CARD }, line: { color: BORDER, width: 1 } });
    s.addText([
      { text: "工程方法论：  ", options: { bold: true, color: NAVY } },
      { text: "样本 ≥ 30 才下结论  ·  防 look-ahead bias  ·  参数改动必有回测依据  ·  结论为负也如实接受", options: { color: INK } },
    ], { x: 1.04, y: 4.56, w: 8.3, h: 0.5, margin: 0, valign: "middle", fontFace: BF, fontSize: 10.3 });

    footer(s, 9);
  }

  // ════════════════════════════════════════════════════
  //  Slide 10 — 回测引擎
  // ════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    header(s, "Backtest Engine", "回测引擎 — 贴近实盘的模拟");

    // 左：交易规则建模
    s.addText("真实交易规则建模", {
      x: 0.86, y: 1.42, w: 4, h: 0.34, margin: 0,
      fontFace: HF, fontSize: 14, bold: true, color: NAVY,
    });
    const rules = [
      "T+1 — 当日买入次日才可卖",
      "涨跌停无法成交 — 按板块/ST 阈值判定",
      "次日开盘价成交 — 更接近实盘体验",
      "完整费用 — 佣金万0.86 + 印花税0.1% + 最低5元",
      "多种出场 — 硬止损 / 固定止盈 / 跟踪止盈",
    ];
    s.addText(
      rules.map((t, i) => ({ text: t, options: { bullet: { code: "2022", indent: 12 }, breakLine: i < rules.length - 1, color: INK } })),
      { x: 0.9, y: 1.84, w: 4.5, h: 2.6, margin: 0, fontFace: BF, fontSize: 11.5, paraSpaceAfter: 9 }
    );

    // 右：能力卡
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 5.6, y: 1.42, w: 3.88, h: 3.0, rectRadius: 0.08,
      fill: { color: CARD }, line: { color: BORDER, width: 1 }, shadow: makeShadow(),
    });
    s.addImage({ data: ic.cogs, x: 5.84, y: 1.64, w: 0.4, h: 0.4 });
    s.addText("回测能力", {
      x: 6.34, y: 1.62, w: 3, h: 0.42, margin: 0, valign: "middle",
      fontFace: HF, fontSize: 15, bold: true, color: NAVY,
    });
    const caps = [
      ["单标的引擎", "事件驱动，支持进阶出场策略"],
      ["共振组合回测", "全市场扫描 + 历史信号复现"],
      ["绩效指标", "收益 / 胜率 / 回撤 / 夏普 / 盈亏比"],
    ];
    let ky = 2.24;
    caps.forEach(([k, v]) => {
      s.addShape(pres.shapes.RECTANGLE, { x: 5.86, y: ky + 0.03, w: 0.12, h: 0.12, fill: { color: AMBER } });
      s.addText(k, {
        x: 6.1, y: ky - 0.08, w: 3.2, h: 0.3, margin: 0,
        fontFace: BF, fontSize: 12, bold: true, color: NAVY,
      });
      s.addText(v, {
        x: 6.1, y: ky + 0.2, w: 3.2, h: 0.3, margin: 0,
        fontFace: BF, fontSize: 10, color: MUTED,
      });
      ky += 0.72;
    });

    footer(s, 10);
  }

  // ════════════════════════════════════════════════════
  //  Slide 11 — 工程能力总结
  // ════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: WHITE };
    header(s, "Summary", "这个项目体现的工程能力");

    const skills = [
      ["系统设计", "六层解耦架构，单层职责清晰、可独立替换"],
      ["性能调优", "定位 I/O 瓶颈，批量化 + 索引 + WAL，实测百倍提速"],
      ["健壮性工程", "容错链 / 自愈 / 边界处理 — 覆盖非正常路径"],
      ["数据工程", "多源整合、增量更新、时效性管理、728 万行规模"],
      ["数据驱动决策", "A/B 回测验证假设，尊重数据、不凭直觉"],
      ["可维护性", ".env 秘密管理、SQL 参数化、配置外置、日志规范"],
    ];
    const CW = 4.32, CH = 1.0, GX = 0.22, GY = 0.14, X0 = 0.86, Y0 = 1.42;
    skills.forEach(([k, v], i) => {
      const x = X0 + (i % 2) * (CW + GX);
      const y = Y0 + Math.floor(i / 2) * (CH + GY);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x, y, w: CW, h: CH, rectRadius: 0.07,
        fill: { color: CARD }, line: { color: BORDER, width: 1 }, shadow: makeShadow(),
      });
      s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: y + 0.27, w: 0.46, h: 0.46, fill: { color: AMBER } });
      s.addText(String(i + 1), {
        x: x + 0.2, y: y + 0.27, w: 0.46, h: 0.46, margin: 0, align: "center", valign: "middle",
        fontFace: HF, fontSize: 17, bold: true, color: NAVY_DARK,
      });
      s.addText(k, {
        x: x + 0.8, y: y + 0.14, w: CW - 1.0, h: 0.36, margin: 0,
        fontFace: HF, fontSize: 13.5, bold: true, color: NAVY,
      });
      s.addText(v, {
        x: x + 0.8, y: y + 0.47, w: CW - 1.0, h: 0.46, margin: 0,
        fontFace: BF, fontSize: 10, color: INK, lineSpacingMultiple: 1.12,
      });
    });

    footer(s, 11);
  }

  // ════════════════════════════════════════════════════
  //  Slide 12 — 结尾
  // ════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: NAVY_DARK };

    for (let i = 0; i < 5; i++) {
      s.addShape(pres.shapes.RECTANGLE, {
        x: 8.55 + (i % 3) * 0.34, y: 0.5 + Math.floor(i / 3) * 0.34,
        w: 0.2, h: 0.2, fill: { color: AMBER, transparency: i * 14 },
      });
    }

    s.addText("不止『能跑』，更追求『跑得稳、跑得快、改得动』", {
      x: 0.9, y: 1.7, w: 8.6, h: 1.4, margin: 0,
      fontFace: HF, fontSize: 30, bold: true, color: WHITE, lineSpacingMultiple: 1.15,
    });

    s.addText(
      ["系统设计", "性能调优", "健壮性工程", "数据工程", "数据驱动决策"]
        .map((t, i, arr) => ({ text: t + (i < arr.length - 1 ? "      " : ""), options: { color: i % 2 ? ICE : AMBER, bold: true } })),
      { x: 0.92, y: 3.25, w: 8.6, h: 0.4, margin: 0, fontFace: BF, fontSize: 14 }
    );

    s.addShape(pres.shapes.RECTANGLE, { x: 0.92, y: 3.95, w: 0.5, h: 0.04, fill: { color: AMBER } });
    s.addText("A 股量化交易系统  ·  个人作品集  ·  Python / SQLite / 量化策略", {
      x: 0.92, y: 4.15, w: 8.6, h: 0.34, margin: 0,
      fontFace: BF, fontSize: 12, color: STEEL,
    });

    s.addText("感谢阅读", {
      x: 0.92, y: 4.5, w: 4, h: 0.34, margin: 0,
      fontFace: BF, fontSize: 12, bold: true, color: ICE,
    });
  }

  await pres.writeFile({ fileName: "D:/All code/quant_system/demo/量化系统作品集.pptx" });
  console.log("OK — 量化系统作品集.pptx generated");
})().catch((e) => { console.error(e); process.exit(1); });
