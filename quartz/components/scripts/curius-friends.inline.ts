import { CuriusResponse } from "../types"
import { fetchFollowing, timeSince } from "./curius"
import { registerMouseHover, removeAllChildren } from "./util"

document.addEventListener("nav", async () => {
  const friends = document.getElementById("friends-list") as HTMLUListElement | null
  const seeMoreFriends = document.getElementById("see-more-friends") as HTMLDivElement | null
  if (!friends) return

  // similar to curius.app, get the first 12
  const response = await fetchFollowing().then((res) => res.slice(0, 12))

  removeAllChildren(friends)
  response.map((user, index) => {
    const { user: User, link: Link } = user
    const li = document.createElement("li")
    li.classList.add("friend-li")

    const onClick = () => {
      window.open(Link.link, "_blank")
    }
    li.addEventListener("click", onClick)
    window.addCleanup(() => li.removeEventListener("click", onClick))

    registerMouseHover(li, "focus")

    // only show the first four friends
    index < 4 ? li.classList.add("active") : (li.id = "inactive")

    // title div
    const titleDiv = document.createElement("div")
    titleDiv.classList.add("friend-title")

    const name = document.createElement("a")
    name.classList.add("friend-name")
    name.innerHTML = `${User.firstName} ${User.lastName}`
    name.style.fontWeight = "bold"
    name.href = `https://curius.app/${User.userLink}`
    name.target = "_blank"

    const time = document.createElement("span")
    time.id = `curius-span-${user.link.id}`
    const modifiedDate = new Date(Link.modifiedDate)
    time.innerHTML = `<time datetime=${
      Link.modifiedDate
    } title="${modifiedDate.toUTCString()}">${timeSince(Link.createdDate)}</time>`
    titleDiv.append(name, time)

    // description div
    const descriptionDiv = document.createElement("div")
    descriptionDiv.classList.add("friend-shortcut")
    descriptionDiv.innerHTML = `in <span style="color: var(--gray) !important">${Link.title}</span>`

    li.append(titleDiv, descriptionDiv)

    friends.appendChild(li)
  })

  const onClickSeeMore = () => {
    const ul = document.getElementById("friends-list") as HTMLUListElement | null
    const svgChev = seeMoreFriends?.querySelectorAll("svg")[0] as SVGSVGElement | null
    const moreText = seeMoreFriends?.querySelectorAll("span")[0] as HTMLSpanElement | null
    const showMore = Array.from(ul?.children as Iterable<HTMLLIElement>).filter(
      (li) => li.id === "inactive",
    )
    if (seeMoreFriends?.classList.contains("expand")) {
      seeMoreFriends.classList.remove("expand")
      showMore.map((li) => li.classList.remove("active"))
      if (svgChev) {
        svgChev.classList.remove("fold")
        svgChev.viewBox.baseVal.y = -10
      }
      if (moreText) moreText.textContent = "de plus"
    } else {
      seeMoreFriends?.classList.add("expand")
      showMore.map((li) => li.classList.add("active"))
      if (svgChev) {
        svgChev.classList.add("fold")
        svgChev.viewBox.baseVal.y = 10
      }
      if (moreText) moreText.textContent = "moins"
    }
  }

  seeMoreFriends?.addEventListener("click", onClickSeeMore)
  window.addCleanup(() => seeMoreFriends?.removeEventListener("click", onClickSeeMore))
})
