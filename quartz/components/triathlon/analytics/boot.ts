export const TRI_ANALYTICS_BOOT_CLASS = 'tri-analytics-booting'

export const TRI_ANALYTICS_BOOT_SCRIPT =
  "if (/\\/triathlon\\/analytics\\/?$/.test(window.location.pathname) || (window.location.hostname === 't.aarnphm.xyz' && /^\\/analytics\\/?$/.test(window.location.pathname))) document.documentElement.classList.add('tri-analytics-booting')"
