const fs = require('fs')

const input = fs.readFileSync('./d1.txt', 'utf-8').trim().split('\n')

function p1(i) {
  let position = 50
  let count = 0

  for (const line of i) {
    const direction = line[0]
    const distance = parseInt(line.slice(1), 10)
    if (direction === 'R') {
      position = (position + distance) % 100
    } else {
      position = (((position - distance) % 100) + 100) % 100
    }

    if (position === 0) count++
  }
  return count
}

function zeros(position, direction, distance) {
  if (direction === 'R') {
    return Math.floor((position + distance) / 100)
  } else {
    if (position === 0) {
      return Math.floor(distance / 100)
    }

    return distance >= position ? Math.floor((distance - position) / 100) + 1 : 0
  }
}

function p2(i) {
  let position = 50
  let count = 0

  for (const line of i) {
    const direction = line[0]
    const distance = parseInt(line.slice(1), 10)

    count += zeros(position, direction, distance)
    if (direction === 'R') {
      position = (position + distance) % 100
    } else {
      position = (((position - distance) % 100) + 100) % 100
    }
  }
  return count
}

console.log(`p1: ${p1(input)}`)
console.log(`p2: ${p2(input)}`)
