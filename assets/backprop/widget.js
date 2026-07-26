(function(){
  // --- 高度上报：让宿主页把 iframe 撑到内容实际高度 ---
  function send(){
    var h = Math.ceil(document.documentElement.getBoundingClientRect().height);
    parent.postMessage({backpropWidget:true, id:location.pathname.split('/').pop(), height:h}, '*');
  }
  addEventListener('load', send);
  addEventListener('resize', send);
  if (window.ResizeObserver) new ResizeObserver(send).observe(document.body);
  setTimeout(send, 120); setTimeout(send, 600);

  // --- 主题同步：镜像宿主页（Chirpy 7 / Bootstrap 5.3）<html data-bs-theme> 到 iframe 自身 ---
  // Chirpy 总会把解析后的主题写进该属性（含"跟随系统"）；万一跨域取不到，静默退回 prefers-color-scheme。
  function syncMode(){
    var m = parent.document.documentElement.getAttribute('data-bs-theme');
    if (m) document.documentElement.setAttribute('data-bs-theme', m);
    else document.documentElement.removeAttribute('data-bs-theme');
  }
  try {
    syncMode();
    new MutationObserver(syncMode)
      .observe(parent.document.documentElement, {attributes:true, attributeFilter:['data-bs-theme']});
  } catch(e) {}
})();
