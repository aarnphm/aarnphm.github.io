const phases = [
  "oops! this page is still in the oven! 🍳",
  "feel free to check back later 🥧",
  "or contact me if you have any further questions 🤗",
]

let currentPhraseIndex = 0
const typingSpeed = 50 // Duration in milliseconds
const deletingSpeed = 30 // Duration in milliseconds

const typePhrase = (element: HTMLElement, phrase: string, index: number, callback: () => void) => {
  if (index < phrase.length) {
    element.textContent = phrase.substring(0, index + 1)
    setTimeout(() => typePhrase(element, phrase, index + 1, callback))
  } else {
    setTimeout(callback, 2000) // Wait for 2 seconds at the end of the phrase
  }
}

function deletePhrase(element: HTMLElement, callback: () => void) {
  let currentText = element.textContent ?? ""
  if (currentText.length > 0) {
    element.textContent = currentText.substring(0, currentText.length - 1)
    setTimeout(() => deletePhrase(element, callback), deletingSpeed)
  } else {
    setTimeout(callback, 500) // Wait for half a second before starting the new phrase
  }
}

function startTypingEffect(element: HTMLElement, phrasesArray: string[]) {
  if (currentPhraseIndex >= phrasesArray.length) {
    currentPhraseIndex = 0 // Reset to start again if needed
  }

  typePhrase(element, phrasesArray[currentPhraseIndex], 0, () => {
    deletePhrase(element, () => {
      currentPhraseIndex++
      startTypingEffect(element, phrasesArray)
    })
  })
}

document.addEventListener("nav", () => {
  const typewritter = document.getElementById("typewritter")
  if (!typewritter) return
  typewritter.classList.add("typing")
  startTypingEffect(typewritter, phases)
})
