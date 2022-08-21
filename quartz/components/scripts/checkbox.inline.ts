import { getFullSlug } from "../../util/path"

function setupCheckbox() {
  const checkboxes = document.querySelectorAll(
    "input.checkbox-toggle",
  ) as NodeListOf<HTMLInputElement>
  checkboxes.forEach((el, index) => {
    const elId = `${getFullSlug(window)}-checkbox-${index}`

    function switchState(e: Event) {
      const newCheckboxState = (e.target as HTMLInputElement)?.checked ? "true" : "false"
      localStorage.setItem(elId, newCheckboxState)
    }

    el.addEventListener("change", switchState)
    window.addCleanup(() => el.removeEventListener("change", switchState))
    if (localStorage.getItem(elId) === "true") {
      el.checked = true
    }
  })
}

document.addEventListener("nav", setupCheckbox)
document.addEventListener("content-decrypted", setupCheckbox)
