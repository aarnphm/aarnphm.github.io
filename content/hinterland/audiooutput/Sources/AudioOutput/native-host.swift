import Foundation

struct BridgeRequest: Decodable {
    let type: String
    let host: String?
    let state: String?
    let detail: String?
    let outputUID: String?
}

struct BridgeResponse: Encodable {
    let configuration: SiteConfiguration?
    let error: String?
}

enum NativeHost {
    static func run() throws {
        let input = FileHandle.standardInput
        let output = FileHandle.standardOutput
        while let header = try readExactly(4, from: input) {
            let size = header.withUnsafeBytes { $0.loadUnaligned(as: UInt32.self).littleEndian }
            guard size > 0, size <= 65536, let data = try readExactly(Int(size), from: input) else {
                throw AudioFailure(message: "Invalid native-message length")
            }
            let response: BridgeResponse
            do {
                let request = try JSONDecoder().decode(BridgeRequest.self, from: data)
                switch request.type {
                case "snapshot":
                    response = BridgeResponse(configuration: SiteRouting.snapshot(), error: nil)
                case "subscribe":
                    guard let host = request.host, SiteRouting.validHost(host) else {
                        throw AudioFailure(message: "Invalid site hostname")
                    }
                    var configuration = SiteRouting.load()
                    if !configuration.sites.contains(where: { $0.id == host }) {
                        configuration.sites.append(SiteRule(id: host, name: host, outputUID: "", outputName: "Normal output"))
                        try SiteRouting.save(configuration)
                    }
                    response = BridgeResponse(configuration: SiteRouting.snapshot(), error: nil)
                case "status":
                    guard let host = request.host, let state = request.state, let detail = request.detail,
                          let outputUID = request.outputUID,
                          ["routing", "normal", "permission", "missing", "error", "waiting"].contains(state),
                          detail.count <= 500,
                          SiteRouting.load().sites.contains(where: { $0.id == host }) else {
                        throw AudioFailure(message: "Invalid site status")
                    }
                    try SiteRouting.report(SiteStatus(host: host, state: state, detail: detail,
                                                      outputUID: outputUID, updatedAt: Date()))
                    response = BridgeResponse(configuration: nil, error: nil)
                default:
                    throw AudioFailure(message: "Unknown native-message type")
                }
            } catch { response = BridgeResponse(configuration: nil, error: error.localizedDescription) }
            let encoded = try JSONEncoder().encode(response)
            var length = UInt32(encoded.count).littleEndian
            try withUnsafeBytes(of: &length) { try output.write(contentsOf: $0) }
            try output.write(contentsOf: encoded)
        }
    }

    private static func readExactly(_ count: Int, from input: FileHandle) throws -> Data? {
        var data = Data()
        while data.count < count {
            guard let chunk = try input.read(upToCount: count - data.count), !chunk.isEmpty else {
                if data.isEmpty { return nil }
                throw AudioFailure(message: "Truncated native message")
            }
            data.append(chunk)
        }
        return data
    }
}
