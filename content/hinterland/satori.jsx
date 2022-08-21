// Fallback fonts and Emoji are dynamically loaded
// from Google Fonts and CDNs in this demo.
//
// You can also return a function component in the playground.
// NOTE: make sure to use this within a iife
const cfg = {
  baseUrl: "aarnphm.xyz",
  theme: {
    cdnCaching: true,
    fontOrigin: "googleFonts",
    typography: {
      header: "EB Garamond",
      body: "EB Garamond",
      code: "JetBrains Mono",
    },
    colors: {
      lightMode: {
        light: "#fffaf3",
        lightgray: "#f2e9e1",
        gray: "#9893a5",
        darkgray: "#797593",
        dark: "#575279",
        secondary: "#d7827e",
        tertiary: "#b4637a",
        highlight: "rgba(143, 159, 169, 0.15)",
        textHighlight: "rgba(246, 193, 119, 0.28)",
      },
      darkMode: {
        light: "#1f1d30",
        lightgray: "#26233a",
        gray: "#6e6a86",
        darkgray: "#908caa",
        dark: "#e0def4",
        secondary: "#ebbcba",
        tertiary: "#eb6f92",
        highlight: "rgba(143, 159, 169, 0.15)",
        textHighlight: "#b3aa0288",
      },
    },
  },
}
const url = new URL(`https://${cfg.baseUrl}`)

const fetchFonts = (font) => {
  const res = fetch(`${url.toString()}/${font}`).then((data) => data.arrayBuffer())
  return res
}

const promises = [
  fetchFonts("static/GT-Sectra-Display-Regular.woff"),
  fetchFonts("static/GT-Sectra-Book.woff"),
]

const [header, body] = Promise.all(promises)
const fonts = [
  {
    name: cfg.theme.typography.header,
    data: header,
    weight: 700,
    style: "normal",
  },
  {
    name: cfg.theme.typography.body,
    data: body,
    weight: 400,
    style: "normal",
  },
]
const colorScheme = "lightMode"
const title = "mechanistic interpretability"
const Li = ["Jan 15 2025", "20 min read"]
const description = "reverse engineering models"

return (
  <div
    style={{
      position: "relative",
      display: "flex",
      flexDirection: "row",
      alignItems: "flex-start",
      height: "100%",
      width: "100%",
      backgroundImage: `url("https://${cfg.baseUrl}/static/og-image.jpeg")`,
      backgroundSize: "100% 100%",
    }}
  >
    <div
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: "radial-gradient(circle at center, transparent, rgba(0, 0, 0, 0.4) 70%)",
      }}
    />
    <div
      style={{
        display: "flex",
        height: "100%",
        width: "100%",
        flexDirection: "column",
        justifyContent: "flex-start",
        alignItems: "flex-start",
        gap: "1.5rem",
        paddingTop: "4rem",
        paddingBottom: "4rem",
        marginLeft: "4rem",
      }}
    >
      <img
        src={`https://${cfg.baseUrl}/static/icon.jpeg`}
        style={{
          position: "relative",
          backgroundClip: "border-box",
          borderRadius: "6rem",
        }}
        width={80}
      />
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          textAlign: "left",
          fontFamily: fonts[0].name,
        }}
      >
        <h2
          style={{
            color: cfg.theme.colors[colorScheme].light,
            fontSize: "3rem",
            fontWeight: 700,
            marginRight: "4rem",
            fontFamily: fonts[0].name,
          }}
        >
          {title}
        </h2>
        <ul
          style={{
            color: cfg.theme.colors[colorScheme].gray,
            gap: "1rem",
            fontSize: "1.5rem",
            fontFamily: fonts[1].name,
          }}
        >
          {Li.map((item, index) => {
            if (item) {
              return <li key={index}>{item}</li>
            }
          })}
        </ul>
      </div>
      <p
        style={{
          color: cfg.theme.colors[colorScheme].light,
          fontSize: "1.5rem",
          overflow: "hidden",
          marginRight: "8rem",
          textOverflow: "ellipsis",
          display: "-webkit-box",
          WebkitLineClamp: 7,
          WebkitBoxOrient: "vertical",
          lineClamp: 7,
          fontFamily: fonts[1].name,
        }}
      >
        {description}
      </p>
    </div>
  </div>
)
