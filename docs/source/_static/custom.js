/* Keep the primary sidebar persistently visible on desktop layouts. */
(function () {
  const DESKTOP_QUERY = "(min-width: 960px)";

  function forceSidebarDesktopState() {
    if (!window.matchMedia(DESKTOP_QUERY).matches) {
      return;
    }

    const sidebar = document.getElementById("pst-primary-sidebar");
    if (sidebar) {
      sidebar.classList.remove("hide-on-wide");
      sidebar.classList.remove("no-sidebar");
      sidebar.style.display = "block";
      sidebar.style.visibility = "visible";
      sidebar.style.transform = "none";
      sidebar.style.marginLeft = "0";
    }

    const modal = document.getElementById("pst-primary-sidebar-modal");
    if (modal && modal.open) {
      modal.close();
    }
  }

  function enforceRepeatedly() {
    forceSidebarDesktopState();
    window.requestAnimationFrame(forceSidebarDesktopState);
    setTimeout(forceSidebarDesktopState, 100);
    setTimeout(forceSidebarDesktopState, 350);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", enforceRepeatedly);
  } else {
    enforceRepeatedly();
  }

  window.addEventListener("resize", enforceRepeatedly);
})();
