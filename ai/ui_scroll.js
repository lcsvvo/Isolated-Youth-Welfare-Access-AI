// Fixed application state only: no input text, external requests, or storage.
(() => {
  const state = __SCROLL_STATE__;
  const previous = window.__halfstepScroll;
  if (previous?.token === state.token) return;
  previous?.cancel();
  // Streamlit changes the main element's test id when a pinned chat input is
  // mounted. The semantic element remains the same scroll container.
  const main = document.querySelector("main");
  let frame;
  let cancelled = false;
  const cancel = () => {
    cancelled = true;
    cancelAnimationFrame(frame);
    for (const event of ["wheel", "touchstart", "pointerdown", "keydown"])
      document.removeEventListener(event, cancel, true);
  };
  window.__halfstepScroll = {token: state.token, cancel};
  if (!main || (state.page !== "results" && !state.anchor)) return;
  for (const event of ["wheel", "touchstart", "pointerdown", "keydown"])
    document.addEventListener(event, cancel, {capture: true, passive: true});
  const started = performance.now();
  const settle = () => {
    if (cancelled) return;
    if (state.page === "results") {
      main.scrollTo({top: 0, behavior: "instant"});
    } else if (state.anchor) {
      const anchor = document.getElementById(state.anchor);
      if (anchor) main.scrollTo({
        top: main.scrollTop + anchor.getBoundingClientRect().top - main.getBoundingClientRect().top - 24,
        behavior: "instant",
      });
    }
    // Let Streamlit finish its message layout and native chat-input scrolling.
    // User input cancels immediately, so reading never gets pinned in place.
    if (performance.now() - started < 1200) frame = requestAnimationFrame(settle);
    else cancel();
  };
  frame = requestAnimationFrame(settle);
})();
