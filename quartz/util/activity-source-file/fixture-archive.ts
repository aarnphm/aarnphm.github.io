import { crc32, deflateRawSync } from 'node:zlib'

interface ArchiveMember {
  name: string
  bytes: Uint8Array
  method: 0 | 8
}

export function fixtureArchive(members: readonly ArchiveMember[]): Uint8Array {
  const localParts: Uint8Array[] = []
  const directoryParts: Uint8Array[] = []
  let offset = 0
  for (const member of members) {
    const name = Buffer.from(member.name)
    const compressed = member.method === 8 ? deflateRawSync(member.bytes) : member.bytes
    const checksum = crc32(member.bytes)
    const local = Buffer.alloc(30 + name.length)
    local.writeUInt32LE(0x04034b50, 0)
    local.writeUInt16LE(20, 4)
    local.writeUInt16LE(member.method, 8)
    local.writeUInt32LE(checksum, 14)
    local.writeUInt32LE(compressed.length, 18)
    local.writeUInt32LE(member.bytes.length, 22)
    local.writeUInt16LE(name.length, 26)
    name.copy(local, 30)
    localParts.push(local, compressed)
    const central = Buffer.alloc(46 + name.length)
    central.writeUInt32LE(0x02014b50, 0)
    central.writeUInt16LE(20, 4)
    central.writeUInt16LE(20, 6)
    central.writeUInt16LE(member.method, 10)
    central.writeUInt32LE(checksum, 16)
    central.writeUInt32LE(compressed.length, 20)
    central.writeUInt32LE(member.bytes.length, 24)
    central.writeUInt16LE(name.length, 28)
    central.writeUInt32LE(offset, 42)
    name.copy(central, 46)
    directoryParts.push(central)
    offset += local.length + compressed.length
  }
  const directory = Buffer.concat(directoryParts)
  const end = Buffer.alloc(22)
  end.writeUInt32LE(0x06054b50, 0)
  end.writeUInt16LE(members.length, 8)
  end.writeUInt16LE(members.length, 10)
  end.writeUInt32LE(directory.length, 12)
  end.writeUInt32LE(offset, 16)
  return Buffer.concat([...localParts, directory, end])
}
