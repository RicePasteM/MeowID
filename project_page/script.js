if (window.lucide) {
  window.lucide.createIcons();
}

const chineseTranslations = {
  skipContent: '跳到主要内容',
  announcement: 'ICW 基准数据集收录 82,791 张图像，覆盖 19,877 只猫。',
  datasetDetails: '数据集详情',
  primaryNavigation: '主导航',
  meowidHome: 'MeowID 首页',
  toggleMenu: '切换菜单',
  navMethod: '方法',
  navDataset: '数据集',
  navResults: '结果',
  readPaper: '阅读论文',
  readPaperShort: '阅读论文',
  code: '代码',
  languageSelector: '语言选择',
  teaserField: '可交互的猫个体观测区域',
  heroTitle1: '面向猫个体识别的',
  heroTitle2: '双专家检索系统',
  authors: '<strong>胡张驰</strong>、尚艺、杨皓程、胡琪伟、李煜政',
  authorAffiliations: '作者单位',
  affiliationUstc: '中国科学技术大学',
  affiliationHfut: '合肥工业大学',
  affiliationNwpu: '西北工业大学',
  affiliationBjfu: '北京林业大学',
  affiliationSysu: '中山大学',
  keyResults: '主要结果',
  metricTop1: 'Top-1 准确率',
  metricIcwSet: 'ICW 完整测试集',
  metricMap: '检索 mAP',
  metricUnseen: '测试阶段未见个体',
  metricAuc: '平均 AUC',
  metricFamilies: 'PetFace 的 13 类动物',
  metricLatency: '端到端延迟',
  whyMeowid: '为什么选择 MeowID',
  introTitle: '融合不同视觉尺度的互补线索。',
  introLead: '成像条件的变化会显著影响个体外观，单一表征难以捕捉全部身份特征。',
  introBody: 'MeowID 以两个独立专家分别建模面部细节和整猫外观。面部对齐成功时，系统优先使用面部特征；整猫外观用于补充上下文，并在面部不可用时承担检索任务。',
  principleFace: '面部优先路由',
  principleOpen: '开放集增量注册',
  principleGallery: '按路径划分图库',
  methodTitle: '面部优先，缺失时稳健回退。',
  methodLead: '系统既保留细粒度面部判别能力，也能应对无约束图像中的复杂情况。',
  routingBehavior: '检索路径',
  selectFaceAvailability: '选择面部是否可用',
  routeFaceCopy: '面部可用时，系统以面部特征为主，融合整猫信息后进入面部图库检索。',
  routeBodyCopy: '无法获得可靠的对齐面部时，系统改由整猫专家直接查询整猫图库。',
  faceAvailable: '面部可用',
  faceUnavailable: '面部不可用',
  chooseFaceAvailability: '选择面部是否可用',
  interactiveFlow: 'MeowID 交互式检索流程',
  flowConnections: '检索各阶段之间的数据流向',
  routeFaceAria: '面部可用检索路径',
  routeBodyAria: '面部不可用备用路径',
  nodeInput: '输入图像',
  nodeInputNote: '无约束图像',
  nodeCrop: '对齐后的面部',
  nodeCropNote: '局部特征',
  nodeFaceExpert: '面部专家',
  nodeFaceExpertNote: '细粒度面部细节',
  nodeBodyExpert: '整猫专家',
  nodeBodyExpertNote: '全局外观',
  nodeFusion: '整猫提示融合',
  nodeFusionNote: '整猫信息校正',
  nodeFaceGallery: '面部图库',
  nodeFaceGalleryNote: '面部路径检索',
  nodeBodyGallery: '整猫图库',
  nodeBodyGalleryNote: '备用路径检索',
  nodeRanking: '身份排序',
  nodeRankingNote: '按路径输出结果',
  cardLocalization: '面部定位',
  cardLocalizationBody: '利用九个关键点校正平移、平面内旋转和尺度，生成 256 × 256 的对齐面部图像。',
  cardExperts: '独立专家',
  cardExpertsBody: '两个独立的 DINOv3 ViT-B/16 编码器分别生成 512 维面部嵌入和整猫嵌入。',
  cardFusion: '门控上下文融合',
  cardFusionBody: '样本自适应门控在保留面部表征的同时，引入整猫上下文。',
  cardRetrieval: '图库检索',
  cardRetrievalBody: '系统按相似度对身份排序；新增个体可直接注册，无需重新训练模型。',
  datasetName: '自然场景猫个体数据集（ICW）',
  datasetTitle: '面向真实场景猫个体检索的大规模基准数据集。',
  datasetBody: 'ICW 汇集自六个公开的领养与救助平台，并按身份严格划分数据。每只猫包含多次观测，覆盖姿态、视角、背景、光照、拍摄设备和遮挡等真实变化。',
  viewHuggingFace: '前往 Hugging Face',
  statIdentities: '只猫',
  statImages: '张高质量图像',
  statFaceAvailable: '面部可用',
  statImagesPerIdentity: '每只猫的图像数',
  figure02: '图 2',
  icwPipeline: 'ICW 构建流程',
  scrollableFigure: '可横向浏览的 ICW 构建流程图',
  pipelineAlt: 'ICW 数据集的五阶段构建流程',
  pipelineCaption: '公开档案依次经过筛选、分割、去重和人工复核，随后按身份划分数据，确保训练集与评估集不存在身份重叠。',
  resultsTitle: 'Top-1 检索性能进一步提升。',
  resultsLead: '在完整 ICW 测试集上，MeowID 取得最佳 Top-1 准确率和 mAP；对于未检测到面部的查询，系统仍能返回有效结果。',
  icwFullSet: 'ICW 完整测试集',
  top1Accuracy: 'Top-1 检索准确率',
  best: '最佳',
  top1Comparison: 'Top-1 准确率对比',
  top1Note: '相较 PetFace，Top-1 准确率提高 4.07 个百分点。',
  faceRoute: '面部路径',
  hintTitle: '整猫上下文进一步改善面部检索。',
  faceOnly: '仅面部',
  faceHint: '面部 + 整猫提示',
  improved: '提升',
  unchanged: '不变',
  degraded: '下降',
  rankNote: '加入整猫提示后，537 个查询的真实身份排名上升，82 个查询的排名下降。',
  fullTestSet: '完整 ICW 测试集',
  retrievalComparison: '检索性能对比',
  querySummary: '2,846 次查询 · 500 个测试阶段未见个体',
  model: '模型',
  beyondCats: '不止于猫',
  generalizationTitle: '泛化至<br />13 类动物。',
  generalizationBody: '在 PetFace 验证任务上，MeowID 的平均 AUC 达到 94.94%，并在全部 13 类动物上超过对应的 PetFace-ArcFace 专用基线。',
  aucAria: '平均 AUC 为 94.94%',
  avgAuc: '平均 AUC',
  deployment: '部署',
  deploymentTitle: '端到端推理仅需 60 ms。',
  deploymentBody: '延迟包含图像解码、预处理、面部定位与对齐、特征提取以及路径选择。',
  citation: '引用',
  citationTitle: '引用 MeowID。',
  citationBody: '如果本研究对您的工作有所帮助，请引用以下条目。',
  copyBibtex: '复制 BibTeX',
  copied: '已复制',
  selected: '已选中',
  openResearch: '面向动物个体识别的开放研究',
  finalTitle: '突破单一视角的<br />个体检索。',
  exploreCode: '查看代码',
  footerDescription: '面向真实场景猫个体识别的双专家检索系统。',
  paper: '论文',
  copyright: '© 2026 MeowID 研究团队',
  footerTagline: '面向真实场景观测的个体检索。'
};

const englishFallbacks = {
  routeBodyCopy: 'Without an aligned face, the whole-cat expert retrieves directly from the whole-cat gallery.',
  routeFaceAria: 'Face-available retrieval route',
  routeBodyAria: 'Face-unavailable fallback route',
  copied: 'Copied!',
  selected: 'Selected'
};

const englishContent = new Map();
document.querySelectorAll('[data-i18n]').forEach((element) => {
  if (!englishContent.has(element.dataset.i18n)) englishContent.set(element.dataset.i18n, element.textContent.trim());
});
const englishHtml = new Map();
document.querySelectorAll('[data-i18n-html]').forEach((element) => {
  if (!englishHtml.has(element.dataset.i18nHtml)) englishHtml.set(element.dataset.i18nHtml, element.innerHTML.trim());
});
const englishAria = new Map();
document.querySelectorAll('[data-i18n-aria]').forEach((element) => {
  if (!englishAria.has(element.dataset.i18nAria)) englishAria.set(element.dataset.i18nAria, element.getAttribute('aria-label'));
});
const englishAlt = new Map();
document.querySelectorAll('[data-i18n-alt]').forEach((element) => {
  if (!englishAlt.has(element.dataset.i18nAlt)) englishAlt.set(element.dataset.i18nAlt, element.getAttribute('alt'));
});

let storedLanguage;
try { storedLanguage = window.localStorage.getItem('meowid-language'); } catch { storedLanguage = null; }
let currentLanguage = storedLanguage === 'zh-CN' || storedLanguage === 'en'
  ? storedLanguage
  : (navigator.language.toLowerCase().startsWith('zh') ? 'zh-CN' : 'en');

const translate = (key, source = englishContent) => currentLanguage === 'zh-CN'
  ? (chineseTranslations[key] ?? source.get(key) ?? englishFallbacks[key] ?? key)
  : (source.get(key) ?? englishFallbacks[key] ?? key);

const applyLanguage = (language, persist = true) => {
  currentLanguage = language === 'zh-CN' ? 'zh-CN' : 'en';
  document.documentElement.lang = currentLanguage;
  document.documentElement.dataset.language = currentLanguage;

  document.querySelectorAll('[data-i18n]').forEach((element) => {
    element.textContent = translate(element.dataset.i18n);
  });
  document.querySelectorAll('[data-i18n-html]').forEach((element) => {
    element.innerHTML = translate(element.dataset.i18nHtml, englishHtml);
  });
  document.querySelectorAll('[data-i18n-aria]').forEach((element) => {
    element.setAttribute('aria-label', translate(element.dataset.i18nAria, englishAria));
  });
  document.querySelectorAll('[data-i18n-alt]').forEach((element) => {
    element.setAttribute('alt', translate(element.dataset.i18nAlt, englishAlt));
  });

  document.title = currentLanguage === 'zh-CN'
    ? 'MeowID — 面向猫个体识别的双专家检索系统'
    : 'MeowID — A Dual-Expert Retrieval System for Individual Cat Identification';
  document.querySelector('meta[name="description"]').content = currentLanguage === 'zh-CN'
    ? 'MeowID 是一套面向真实场景猫个体识别的双专家检索系统。'
    : 'MeowID is a dual-expert retrieval system for individual cat identification in unconstrained photographs.';

  document.querySelectorAll('.language-switcher [data-language]').forEach((button) => {
    const active = button.dataset.language === currentLanguage;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });

  document.querySelectorAll('.teaser-cat img').forEach((image, index) => {
    image.alt = currentLanguage === 'zh-CN' ? `ICW 猫咪观测图 ${index + 1}` : `ICW cat observation ${index + 1}`;
  });

  const activeRouteMap = document.querySelector('.route-map');
  const activeRouteCopy = document.querySelector('.route-selection-copy');
  if (activeRouteMap && activeRouteCopy) {
    const faceRoute = activeRouteMap.dataset.activeRoute === 'face';
    activeRouteMap.setAttribute('aria-label', translate(faceRoute ? 'routeFaceAria' : 'routeBodyAria'));
    activeRouteCopy.textContent = translate(faceRoute ? 'routeFaceCopy' : 'routeBodyCopy');
  }

  if (persist) {
    try { window.localStorage.setItem('meowid-language', currentLanguage); } catch { /* Storage can be unavailable on local files. */ }
  }
};

document.querySelectorAll('.language-switcher [data-language]').forEach((button) => {
  button.addEventListener('click', () => {
    applyLanguage(button.dataset.language);
    document.querySelector('.nav-links')?.classList.remove('open');
    document.querySelector('.menu-button')?.setAttribute('aria-expanded', 'false');
  });
});

applyLanguage(currentLanguage, false);

const header = document.querySelector('.site-header');
const menuButton = document.querySelector('.menu-button');
const navLinks = document.querySelector('.nav-links');

window.addEventListener('scroll', () => {
  header.classList.toggle('scrolled', window.scrollY > 16);
}, { passive: true });

menuButton.addEventListener('click', () => {
  const isOpen = navLinks.classList.toggle('open');
  menuButton.setAttribute('aria-expanded', String(isOpen));
});

navLinks.querySelectorAll('a').forEach((link) => {
  link.addEventListener('click', () => {
    navLinks.classList.remove('open');
    menuButton.setAttribute('aria-expanded', 'false');
  });
});

const revealObserver = new IntersectionObserver((entries, observer) => {
  entries.forEach((entry) => {
    if (!entry.isIntersecting) return;
    const delay = entry.target.dataset.delay || 0;
    entry.target.style.setProperty('--delay', `${delay}ms`);
    entry.target.classList.add('visible');
    observer.unobserve(entry.target);
  });
}, { threshold: 0.12, rootMargin: '0px 0px -40px' });

document.querySelectorAll('.reveal').forEach((element) => revealObserver.observe(element));

const routeMap = document.querySelector('.route-map');
const routeButtons = document.querySelectorAll('.route-toggle');
const routeSelectionCopy = document.querySelector('.route-selection-copy');

if (routeMap) {
  const svgNamespace = 'http://www.w3.org/2000/svg';
  const particleSpacing = 20;
  const particleSpeed = 72;

  routeMap.querySelectorAll('.flow-connection').forEach((connection) => {
    const guide = connection.querySelector('.flow-light');
    const pathData = guide?.getAttribute('d');
    if (!guide || !pathData) return;

    const pathLength = guide.getTotalLength();
    const particleCount = Math.max(2, Math.ceil(pathLength / particleSpacing));
    const duration = Math.max(.75, pathLength / particleSpeed);

    for (let index = 0; index < particleCount; index += 1) {
      const particle = document.createElementNS(svgNamespace, 'circle');
      const motion = document.createElementNS(svgNamespace, 'animateMotion');
      particle.classList.add('flow-particle');
      particle.setAttribute('r', '2.4');
      particle.setAttribute('aria-hidden', 'true');
      motion.setAttribute('path', pathData);
      motion.setAttribute('dur', `${duration}s`);
      motion.setAttribute('begin', `${-(duration * index / particleCount)}s`);
      motion.setAttribute('repeatCount', 'indefinite');
      particle.appendChild(motion);
      connection.appendChild(particle);
    }
  });
}

routeButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const route = button.dataset.route;
    routeMap.dataset.activeRoute = route;
    routeMap.setAttribute('aria-label', translate(route === 'face' ? 'routeFaceAria' : 'routeBodyAria'));
    routeButtons.forEach((item) => {
      const active = item === button;
      item.classList.toggle('active', active);
      item.setAttribute('aria-pressed', String(active));
    });
    routeSelectionCopy.textContent = translate(route === 'face' ? 'routeFaceCopy' : 'routeBodyCopy');
    routeMap.classList.add('is-switching');
    window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => routeMap.classList.remove('is-switching'));
    });
  });
});

const identityDemo = document.querySelector('.identity-demo');

if (identityDemo) {
  const demoCases = [
    {
      name: 'Miso', view: 'Frontal face', image: 'assets/hero-query-01.png', alt: 'Selected orange-and-white cat query',
      face: 91, body: 68, fusion: 94, fusionLabel: 'Validation-guided fusion',
      id: 'ICW-0142', similarity: '93.8% similarity', route: 'Face route', note: 'Whole-cat hint active', routeKey: 'face'
    },
    {
      name: 'Ash', view: 'Profile variation', image: 'assets/hero-query-02.png', alt: 'Selected gray cat query',
      face: 77, body: 84, fusion: 91, fusionLabel: 'Context-strengthened fusion',
      id: 'ICW-0871', similarity: '91.2% similarity', route: 'Face route', note: 'Global context strengthened', routeKey: 'face'
    },
    {
      name: 'Maple', view: 'Face occluded', image: 'assets/hero-query-03.png', alt: 'Selected sleeping cat query with an occluded face',
      face: 12, body: 89, fusion: 89, fusionLabel: 'Direct whole-cat retrieval',
      id: 'ICW-1206', similarity: '88.7% similarity', route: 'Whole-cat fallback', note: 'No usable face required', routeKey: 'body'
    }
  ];

  const queryImage = document.querySelector('#demo-query-image');
  const matchImage = document.querySelector('#demo-match-image');
  const queryName = document.querySelector('#demo-query-name');
  const queryView = document.querySelector('#demo-query-view');
  const faceMeter = document.querySelector('#demo-face-meter');
  const bodyMeter = document.querySelector('#demo-body-meter');
  const faceScore = document.querySelector('#demo-face-score');
  const bodyScore = document.querySelector('#demo-body-score');
  const fusionLabel = document.querySelector('#demo-fusion-label');
  const fusionScore = document.querySelector('#demo-fusion-score');
  const matchId = document.querySelector('#demo-match-id');
  const matchScore = document.querySelector('#demo-match-score');
  const routeName = document.querySelector('#demo-route-name');
  const routeNote = document.querySelector('#demo-route-note');
  const queryOptions = document.querySelectorAll('.query-option');
  let updateTimer;

  const updateDemo = (index) => {
    const item = demoCases[index];
    window.clearTimeout(updateTimer);
    identityDemo.classList.add('is-updating');
    identityDemo.dataset.route = item.routeKey;

    updateTimer = window.setTimeout(() => {
      queryImage.src = item.image;
      queryImage.alt = item.alt;
      matchImage.src = item.image;
      queryName.textContent = item.name;
      queryView.textContent = item.view;
      faceMeter.style.setProperty('--signal', `${item.face}%`);
      bodyMeter.style.setProperty('--signal', `${item.body}%`);
      faceScore.textContent = (item.face / 100).toFixed(2);
      bodyScore.textContent = (item.body / 100).toFixed(2);
      fusionLabel.textContent = item.fusionLabel;
      fusionScore.textContent = `Fused confidence · ${(item.fusion / 100).toFixed(2)}`;
      matchId.textContent = item.id;
      matchScore.textContent = item.similarity;
      routeName.textContent = item.route;
      routeNote.textContent = item.note;
      queryOptions.forEach((button, buttonIndex) => {
        const active = buttonIndex === index;
        button.classList.toggle('active', active);
        button.setAttribute('aria-pressed', String(active));
      });
      identityDemo.classList.remove('is-updating');
    }, 140);
  };

  queryOptions.forEach((button) => {
    button.addEventListener('click', () => updateDemo(Number(button.dataset.query)));
  });

  if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    identityDemo.addEventListener('pointermove', (event) => {
      const rect = identityDemo.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width;
      const y = (event.clientY - rect.top) / rect.height;
      identityDemo.style.setProperty('--ry', `${(x - .5) * 4}deg`);
      identityDemo.style.setProperty('--rx', `${(.5 - y) * 4}deg`);
      identityDemo.style.setProperty('--mx', `${x * 100}%`);
      identityDemo.style.setProperty('--my', `${y * 100}%`);
    });
    identityDemo.addEventListener('pointerleave', () => {
      identityDemo.style.setProperty('--ry', '0deg');
      identityDemo.style.setProperty('--rx', '0deg');
      identityDemo.style.setProperty('--mx', '50%');
      identityDemo.style.setProperty('--my', '50%');
    });
  }
}

const teaserField = document.querySelector('.teaser-field');

if (teaserField) {
  const teaserCats = [...teaserField.querySelectorAll('.teaser-cat')];
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const catStates = new Map();
  let fieldRect = teaserField.getBoundingClientRect();
  let previousTime = performance.now();
  let resizeTimer;

  const showIdentity = (cat) => {
    const identity = cat.dataset.identity;
    teaserField.classList.add('has-hover');
    teaserCats.forEach((item) => {
      item.classList.toggle('is-related', item.dataset.identity === identity);
      item.classList.toggle('is-hovered', item === cat);
    });
  };

  const clearIdentity = () => {
    if (teaserCats.some((item) => item.classList.contains('is-dragging'))) return;
    teaserField.classList.remove('has-hover');
    teaserCats.forEach((item) => item.classList.remove('is-related', 'is-hovered'));
  };

  const clampToField = (value, size, limit) => {
    const edge = Math.max(24, size / 2);
    return Math.min(limit - edge, Math.max(edge, value));
  };

  const overlaps = (first, second, gap = 0) => first.left < second.right + gap
    && first.right > second.left - gap
    && first.top < second.bottom + gap
    && first.bottom > second.top - gap;

  const placeCatsRandomly = ({ preserveMoved = false, retry = 0 } = {}) => {
    fieldRect = teaserField.getBoundingClientRect();
    const activeCats = teaserCats.filter((cat) => cat.offsetWidth > 0 && cat.offsetHeight > 0);
    const entries = activeCats.map((cat) => {
      const bounds = cat.getBoundingClientRect();
      return { cat, state: catStates.get(cat), width: bounds.width, height: bounds.height };
    });

    const obstaclePadding = window.innerWidth <= 760 ? 10 : 22;
    const motionClearance = reducedMotion ? 2 : (window.innerWidth <= 760 ? 5 : 10);
    const obstacleSelectors = [
      '.meowid-hub',
      '.hero-center h1',
      '.hero-center .hero-lead',
      '.hero-center .hero-actions',
      '.hero-center .hero-credits'
    ];
    const heroCenter = document.querySelector('.hero-center');
    const heroTransform = heroCenter ? getComputedStyle(heroCenter).transform : 'none';
    const heroMatrix = heroTransform === 'none' ? null : new DOMMatrixReadOnly(heroTransform);
    const revealOffsetX = heroMatrix?.m41 || 0;
    const revealOffsetY = heroMatrix?.m42 || 0;
    const obstacles = obstacleSelectors.map((selector) => document.querySelector(selector))
      .filter((element) => element && element.offsetWidth > 0 && element.offsetHeight > 0)
      .map((element) => {
        const rect = element.getBoundingClientRect();
        return {
          left: rect.left - fieldRect.left - revealOffsetX - obstaclePadding,
          right: rect.right - fieldRect.left - revealOffsetX + obstaclePadding,
          top: rect.top - fieldRect.top - revealOffsetY - obstaclePadding,
          bottom: rect.bottom - fieldRect.top - revealOffsetY + obstaclePadding
        };
      });
    const placed = [];

    const commit = (entry, x, y) => {
      const { cat, state, width, height } = entry;
      state.x = x;
      state.y = y;
      state.anchorX = x;
      state.anchorY = y;
      state.targetX = x;
      state.targetY = y;
      state.vx = 0;
      state.vy = 0;
      cat.style.setProperty('--tx', `${x.toFixed(2)}px`);
      cat.style.setProperty('--ty', `${y.toFixed(2)}px`);
      placed.push({
        left: x - width / 2,
        right: x + width / 2,
        top: y - height / 2,
        bottom: y + height / 2,
        centerX: x,
        centerY: y
      });
    };

    if (preserveMoved) {
      entries.filter(({ state }) => state.moved).forEach((entry) => {
        const x = clampToField(entry.state.x, entry.width, fieldRect.width);
        const y = clampToField(entry.state.y, entry.height, fieldRect.height);
        commit(entry, x, y);
      });
    }

    const pending = entries.filter(({ state }) => !preserveMoved || !state.moved)
      .sort((a, b) => (b.width * b.height) - (a.width * a.height) || Math.random() - .5);
    const gapTiers = window.innerWidth <= 760 ? [8, 7, 6] : [18, 14, 10];

    pending.forEach((entry) => {
      const halfWidth = entry.width / 2;
      const halfHeight = entry.height / 2;
      const edgePadding = window.innerWidth <= 760 ? 4 : 8;
      let selected = null;

      for (const gap of gapTiers) {
        let best = null;
        const attempts = activeCats.length > 20 ? 2600 : 1500;

        for (let attempt = 0; attempt < attempts; attempt += 1) {
          const x = edgePadding + halfWidth + Math.random() * Math.max(1, fieldRect.width - (edgePadding + halfWidth) * 2);
          const y = edgePadding + halfHeight + Math.random() * Math.max(1, fieldRect.height - (edgePadding + halfHeight) * 2);
          const candidate = {
            left: x - halfWidth,
            right: x + halfWidth,
            top: y - halfHeight,
            bottom: y + halfHeight,
            centerX: x,
            centerY: y
          };

          if (obstacles.some((obstacle) => overlaps(candidate, obstacle, motionClearance))) continue;
          if (placed.some((other) => overlaps(candidate, other, gap))) continue;

          const nearest = placed.length
            ? Math.min(...placed.map((other) => Math.hypot(x - other.centerX, y - other.centerY)))
            : Math.min(fieldRect.width, fieldRect.height) * .25;
          const score = nearest + Math.random() * Math.min(fieldRect.width, fieldRect.height) * .08;
          if (!best || score > best.score) best = { x, y, score };
        }

        if (best) {
          selected = best;
          break;
        }
      }

      if (!selected) {
        const scanStep = 4;
        const startX = halfWidth + edgePadding + Math.random() * scanStep;
        const startY = halfHeight + edgePadding + Math.random() * scanStep;
        for (let y = startY; y <= fieldRect.height - halfHeight - edgePadding && !selected; y += scanStep) {
          for (let x = startX; x <= fieldRect.width - halfWidth - edgePadding; x += scanStep) {
            const candidate = { left: x - halfWidth, right: x + halfWidth, top: y - halfHeight, bottom: y + halfHeight };
            if (obstacles.some((obstacle) => overlaps(candidate, obstacle, motionClearance))) continue;
            if (placed.some((other) => overlaps(candidate, other, gapTiers.at(-1)))) continue;
            selected = { x, y };
            break;
          }
        }
      }

      if (selected) commit(entry, selected.x, selected.y);
    });

    if (placed.length !== activeCats.length && retry < 5) {
      placeCatsRandomly({ preserveMoved, retry: retry + 1 });
      return;
    }

    teaserField.dataset.layoutStatus = placed.length === activeCats.length ? 'ready' : 'constrained';
  };

  teaserCats.forEach((cat, index) => {
    const style = getComputedStyle(cat);
    const xPercent = parseFloat(style.getPropertyValue('--x')) / 100;
    const yPercent = parseFloat(style.getPropertyValue('--y')) / 100;
    const state = {
      x: fieldRect.width * xPercent,
      y: fieldRect.height * yPercent,
      anchorX: fieldRect.width * xPercent,
      anchorY: fieldRect.height * yPercent,
      targetX: fieldRect.width * xPercent,
      targetY: fieldRect.height * yPercent,
      vx: 0,
      vy: 0,
      rotation: 0,
      rotationVelocity: 0,
      pointerVelocityX: 0,
      lastPointerX: 0,
      lastPointerTime: 0,
      dragging: false,
      moved: false,
      pointerId: null,
      phaseX: Math.random() * Math.PI * 2,
      phaseY: Math.random() * Math.PI * 2,
      speedX: .16 + Math.random() * .13,
      speedY: .13 + Math.random() * .12,
      amplitudeX: window.innerWidth <= 760 ? 1.25 + Math.random() : 2.5 + Math.random() * 2,
      amplitudeY: window.innerWidth <= 760 ? 1 + Math.random() : 2 + Math.random() * 2,
      index
    };
    catStates.set(cat, state);
    cat.style.setProperty('--tx', `${state.x}px`);
    cat.style.setProperty('--ty', `${state.y}px`);
    cat.style.setProperty('--drag-r', '0deg');

    cat.addEventListener('pointerenter', () => showIdentity(cat));
    cat.addEventListener('pointerleave', () => {
      if (!state.dragging) clearIdentity();
    });
    cat.addEventListener('focus', () => showIdentity(cat));
    cat.addEventListener('blur', clearIdentity);

    cat.addEventListener('pointerdown', (event) => {
      if (event.pointerType === 'mouse' && event.button !== 0) return;
      event.preventDefault();
      fieldRect = teaserField.getBoundingClientRect();
      state.dragging = true;
      state.pointerId = event.pointerId;
      state.targetX = event.clientX - fieldRect.left;
      state.targetY = event.clientY - fieldRect.top;
      state.lastPointerX = event.clientX;
      state.lastPointerTime = event.timeStamp;
      state.pointerVelocityX = 0;
      state.vx *= .35;
      state.vy *= .35;
      cat.classList.add('is-dragging');
      showIdentity(cat);
      cat.setPointerCapture(event.pointerId);
    });

    cat.addEventListener('pointermove', (event) => {
      if (!state.dragging || state.pointerId !== event.pointerId) return;
      const width = cat.offsetWidth || 72;
      const height = cat.offsetHeight || 72;
      const elapsed = Math.max(8, event.timeStamp - state.lastPointerTime);
      const instantaneousVelocity = (event.clientX - state.lastPointerX) / elapsed;
      state.pointerVelocityX = state.pointerVelocityX * .58 + instantaneousVelocity * .42;
      state.lastPointerX = event.clientX;
      state.lastPointerTime = event.timeStamp;
      state.targetX = clampToField(event.clientX - fieldRect.left, width, fieldRect.width);
      state.targetY = clampToField(event.clientY - fieldRect.top, height, fieldRect.height);
    });

    const releaseCat = (event) => {
      if (!state.dragging || state.pointerId !== event.pointerId) return;
      state.dragging = false;
      state.pointerId = null;
      state.anchorX = state.targetX;
      state.anchorY = state.targetY;
      state.pointerVelocityX *= .35;
      state.moved = true;
      cat.classList.remove('is-dragging');
      if (cat.hasPointerCapture(event.pointerId)) cat.releasePointerCapture(event.pointerId);
      if (event.pointerType !== 'mouse') {
        clearIdentity();
        cat.blur();
      }
    };

    cat.addEventListener('pointerup', releaseCat);
    cat.addEventListener('pointercancel', releaseCat);
    cat.addEventListener('click', (event) => event.preventDefault());
  });

  placeCatsRandomly();
  teaserField.classList.add('is-physics-ready');
  if (document.readyState === 'complete') {
    placeCatsRandomly({ preserveMoved: true });
  } else {
    window.addEventListener('load', () => placeCatsRandomly({ preserveMoved: true }), { once: true });
  }

  const animateCats = (time) => {
    const frame = Math.min(2.2, Math.max(.25, (time - previousTime) / 16.667));
    const seconds = time / 1000;
    previousTime = time;

    teaserCats.forEach((cat) => {
      const state = catStates.get(cat);
      const width = cat.offsetWidth || 72;
      const height = cat.offsetHeight || 72;
      let targetX = state.anchorX;
      let targetY = state.anchorY;

      if (state.dragging) {
        targetX = state.targetX;
        targetY = state.targetY;
        state.vx = (state.vx + (targetX - state.x) * .24 * frame) * Math.pow(.68, frame);
        state.vy = (state.vy + (targetY - state.y) * .24 * frame) * Math.pow(.68, frame);
        const targetRotation = Math.max(-30, Math.min(30, (targetX - state.x) * .32 + state.pointerVelocityX * 5));
        state.rotationVelocity = (state.rotationVelocity + (targetRotation - state.rotation) * .22 * frame) * Math.pow(.68, frame);
        state.pointerVelocityX *= Math.pow(.88, frame);
      } else {
        if (!reducedMotion) {
          targetX += Math.sin(seconds * state.speedX + state.phaseX) * state.amplitudeX;
          targetY += Math.cos(seconds * state.speedY + state.phaseY) * state.amplitudeY;
        }
        state.vx = (state.vx + (targetX - state.x) * .008 * frame) * Math.pow(.94, frame);
        state.vy = (state.vy + (targetY - state.y) * .008 * frame) * Math.pow(.94, frame);
        state.rotationVelocity = (state.rotationVelocity - state.rotation * .038 * frame) * Math.pow(.88, frame);
      }

      state.x += state.vx * frame;
      state.y += state.vy * frame;
      state.rotation += state.rotationVelocity * frame;
      if (state.rotation > 30) {
        state.rotation = 30;
        state.rotationVelocity = Math.min(0, state.rotationVelocity);
      } else if (state.rotation < -30) {
        state.rotation = -30;
        state.rotationVelocity = Math.max(0, state.rotationVelocity);
      }

      const boundedX = clampToField(state.x, width, fieldRect.width);
      const boundedY = clampToField(state.y, height, fieldRect.height);
      if (boundedX !== state.x) state.vx *= -.3;
      if (boundedY !== state.y) state.vy *= -.3;
      state.x = boundedX;
      state.y = boundedY;
      cat.style.setProperty('--tx', `${state.x.toFixed(2)}px`);
      cat.style.setProperty('--ty', `${state.y.toFixed(2)}px`);
      cat.style.setProperty('--drag-r', `${state.rotation.toFixed(2)}deg`);
    });

    window.requestAnimationFrame(animateCats);
  };

  window.addEventListener('resize', () => {
    const nextRect = teaserField.getBoundingClientRect();
    const scaleX = nextRect.width / fieldRect.width;
    const scaleY = nextRect.height / fieldRect.height;
    catStates.forEach((state, cat) => {
      state.x *= scaleX;
      state.y *= scaleY;
      state.anchorX *= scaleX;
      state.anchorY *= scaleY;
      state.targetX *= scaleX;
      state.targetY *= scaleY;
    });
    fieldRect = nextRect;
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => placeCatsRandomly({ preserveMoved: true }), 140);
  }, { passive: true });

  window.requestAnimationFrame(animateCats);
}

const copyButton = document.querySelector('#copy-citation');
const bibtex = document.querySelector('#bibtex');

copyButton.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(bibtex.textContent.trim());
    copyButton.textContent = currentLanguage === 'zh-CN' ? chineseTranslations.copied : 'Copied!';
    window.setTimeout(() => { copyButton.textContent = translate('copyBibtex'); }, 1800);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(bibtex);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copyButton.textContent = currentLanguage === 'zh-CN' ? chineseTranslations.selected : 'Selected';
  }
});
