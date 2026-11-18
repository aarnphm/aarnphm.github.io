import Foundation

public enum FITInspector {
  public static func validateActivity(_ data: Data) throws {
    let bytes = [UInt8](data)
    guard !bytes.isEmpty else { throw FITError.truncated }
    var offset = 0
    while offset < bytes.count {
      guard offset + 12 <= bytes.count else { throw FITError.truncated }
      let header = Int(bytes[offset])
      guard header == 12 || header == 14 else { throw FITError.header }
      let length = Int(
        UInt32(bytes[offset + 4]) | UInt32(bytes[offset + 5]) << 8 | UInt32(bytes[offset + 6]) << 16
          | UInt32(bytes[offset + 7]) << 24)
      let end = offset + header + length + 2
      guard end <= bytes.count else { throw FITError.truncated }
      try validateSegment(Data(bytes[offset..<end]), requiresActivity: offset == 0)
      offset = end
    }
  }

  private static func validateSegment(_ data: Data, requiresActivity: Bool) throws {
    let bytes = [UInt8](data)
    guard bytes.count >= 14 else { throw FITError.truncated }
    let header = Int(bytes[0])
    guard header == 12 || header == 14,
      Array(bytes[8..<12]) == [46, 70, 73, 84]
    else { throw FITError.header }
    let length = Int(
      UInt32(bytes[4]) | UInt32(bytes[5]) << 8 | UInt32(bytes[6]) << 16 | UInt32(bytes[7]) << 24)
    guard length > 0, header + length + 2 == bytes.count else { throw FITError.truncated }
    guard crc(bytes) == 0 else { throw FITError.checksum }
    if header == 14, bytes[12] != 0 || bytes[13] != 0 {
      guard crc(Array(bytes[0..<14])) == 0 else { throw FITError.checksum }
    }
    var definitions: [UInt8: Definition] = [:]
    var offset = header
    let end = header + length
    var activity = false
    while offset < end {
      let record = bytes[offset]
      offset += 1
      let compressed = record & 0x80 != 0
      let local = compressed ? (record >> 5) & 3 : record & 15
      if !compressed && record & 0x40 != 0 {
        guard offset + 5 <= end else { throw FITError.truncated }
        let little = bytes[offset + 1] == 0
        let global =
          little
          ? UInt16(bytes[offset + 2]) | UInt16(bytes[offset + 3]) << 8
          : UInt16(bytes[offset + 2]) << 8 | UInt16(bytes[offset + 3])
        let count = Int(bytes[offset + 4])
        offset += 5
        guard offset + count * 3 <= end else { throw FITError.truncated }
        var fields: [(UInt8, Int)] = []
        for _ in 0..<count {
          fields.append((bytes[offset], Int(bytes[offset + 1])))
          offset += 3
        }
        var developerLength = 0
        if record & 0x20 != 0 {
          guard offset < end else { throw FITError.truncated }
          let developers = Int(bytes[offset])
          offset += 1
          guard offset + developers * 3 <= end else { throw FITError.truncated }
          for _ in 0..<developers {
            developerLength += Int(bytes[offset + 1])
            offset += 3
          }
        }
        definitions[local] = Definition(
          global: global, fields: fields, developerLength: developerLength)
      } else {
        guard let definition = definitions[local] else { throw FITError.header }
        for (field, size) in definition.fields {
          if compressed && field == 253 { continue }
          guard offset + size <= end else { throw FITError.truncated }
          if definition.global == 0 && field == 0 && size == 1 {
            guard bytes[offset] == 4 else { throw FITError.notActivity }
            activity = true
          }
          offset += size
        }
        guard offset + definition.developerLength <= end else { throw FITError.truncated }
        offset += definition.developerLength
      }
    }
    guard activity || !requiresActivity else { throw FITError.notActivity }
  }

  private struct Definition {
    let global: UInt16
    let fields: [(UInt8, Int)]
    let developerLength: Int
  }

  private static func crc(_ bytes: [UInt8]) -> UInt16 {
    var value: UInt16 = 0
    for byte in bytes {
      value ^= UInt16(byte)
      for _ in 0..<8 { value = value & 1 == 1 ? (value >> 1) ^ 0xa001 : value >> 1 }
    }
    return value
  }

  public enum FITError: LocalizedError {
    case header, truncated, checksum, notActivity
    public var errorDescription: String? {
      switch self {
      case .header: return "The file has an invalid FIT header or record definition."
      case .truncated: return "The FIT file is incomplete. Read it from the watch again."
      case .checksum: return "The FIT checksum failed. The file was not uploaded."
      case .notActivity: return "This FIT file is not a recorded activity."
      }
    }
  }
}
