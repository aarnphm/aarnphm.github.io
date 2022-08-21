<%\*
// Get the current file's path and content
const currentPath = tp.file.path(true)
const currentFolder = currentPath.substring(0, currentPath.lastIndexOf("/"))
const currentFile = app.vault.getAbstractFileByPath(currentPath)
const fileContent = await app.vault.read(currentFile)

// Define the section markers
const sectionHeader = "<!-- START PDF DATAVIEW -->"
const sectionEnd = "<!-- END PDF DATAVIEW -->"

// Get all PDF files in the vault
const allFiles = app.vault.getFiles()
const pdfFiles = allFiles
.filter((file) => file.extension === "pdf" && file.path.includes(currentFolder) && !file.path.includes(".ignore"))
// Sort alphabetically by filename (case-insensitive)
.sort((a, b) => {
const nameA = a.path.split("/").pop().toLowerCase()
const nameB = b.path.split("/").pop().toLowerCase()
return nameA.localeCompare(nameB)
})

// Create markdown links for each PDF
let output = `${sectionHeader}\n\n`
for (const file of pdfFiles) {
const fileName = file.path.split("/").pop().replace(".pdf", "")
output += `- [[${file.path}|${fileName}]]\n`
}
output += `\n${sectionEnd}`

// Check if the section already exists
const sectionRegex = new RegExp(`${sectionHeader}[\\s\\S]*?${sectionEnd}`)
let updatedContent

if (fileContent.includes(sectionHeader)) {
updatedContent = fileContent.replace(sectionRegex, output)
} else {
updatedContent = fileContent + "\n\n" + output
}

// Write the updated content back to the file
await app.vault.modify(currentFile, updatedContent)

// Return empty string to prevent double insertion
tR = ""
%>
