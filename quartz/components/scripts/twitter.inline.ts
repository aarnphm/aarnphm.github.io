window.twttr = (function (d: Document, s: "script", id: string) {
  var js,
    fjs = d.getElementsByTagName(s)[0],
    t = window.twttr || {}
  if (d.getElementById(id)) return t
  js = d.createElement(s)
  js.id = id
  js.src = "https://platform.twitter.com/widgets.js"
  fjs.parentNode!.insertBefore(js, fjs)

  t._e = []
  t.ready = function (f) {
    t._e.push(f)
  }

  return t
})(document, "script", "twitter-wjs")

document.addEventListener("nav", () => {
  const els = document.querySelectorAll(
    "blockquote.twitter-tweet",
  ) as NodeListOf<HTMLElement> | null
  if (!els || els?.length === 0) return

  for (const el of els) {
    window.twttr.ready((twttr) => twttr.widgets.load(el))
  }
})
