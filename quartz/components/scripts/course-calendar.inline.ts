import { currentNavSignal } from './nav-lifecycle'

type CalendarModule = typeof import('../course-calendar/app')
let calendarModule: Promise<CalendarModule> | undefined

// `course-calendar.js` (quartz.config.ts) loads only on a course calendar page or the French home.
document.addEventListener('nav', () => {
  const root = document.querySelector<HTMLElement>('.course-calendar, .fr-calendar')
  if (!root) return
  const signal = currentNavSignal()
  let unmount: (() => void) | undefined
  window.addCleanup(() => unmount?.())
  calendarModule ??= import(new URL('course-calendar.js', import.meta.url).href)
  void calendarModule
    .then(({ mountCourseCalendar, mountFrCalendar }) => {
      if (signal.aborted || !root.isConnected) return
      unmount = (root.matches('.fr-calendar') ? mountFrCalendar : mountCourseCalendar)(root, signal)
    })
    .catch(error => {
      calendarModule = undefined
      // the build's calendar stays as the page
      if (!signal.aborted) console.error(error)
    })
})
