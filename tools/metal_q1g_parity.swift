// Metal q1g kernel parity, runnable on any Apple-silicon Mac WITHOUT cargo:
//
//   swiftc -O tools/metal_q1g_parity.swift -o /tmp/q1g_parity
//   /tmp/q1g_parity crates/cortiq-engine/src/gpu_metal.rs
//
// It pulls `MSL` (q1_matvec, q1_matvec_h) and `MSL_Q1G` (q1g_matvec,
// q1g_matvec_h) straight out of gpu_metal.rs, compiles each the way the
// engine does (MSL 3.0, default options — fast math on), and runs every
// kernel with the engine's dispatch shape (8 simdgroups × 4 rows per
// threadgroup, weights bound at a nonzero 64-aligned offset like a file
// arena window). The q1 tensor carries ONE f16 scale per 128-group in all
// four of its tiles; the q1g tensor is its repack. q1g must equal q1 bit
// for bit in both the f32 and the half kernel, and match an f64 reference.
// Exit status 0 = all shapes pass.
import Foundation
import Metal

func rawString(_ src: String, _ name: String) -> String {
    let open = "const \(name): &str = r#\""
    guard let a = src.range(of: open), let b = src.range(of: "\"#;", range: a.upperBound..<src.endIndex) else {
        print("cannot find \(name) in the source"); exit(2)
    }
    return String(src[a.upperBound..<b.lowerBound])
}

let args = CommandLine.arguments
guard args.count == 2, let rs = try? String(contentsOfFile: args[1], encoding: .utf8) else {
    print("usage: \(args[0]) crates/cortiq-engine/src/gpu_metal.rs"); exit(2)
}
let dev = MTLCreateSystemDefaultDevice()!
let opts = MTLCompileOptions()
opts.languageVersion = .version3_0
func compile(_ name: String) -> MTLLibrary {
    do { return try dev.makeLibrary(source: rawString(rs, name), options: opts) } catch {
        print("\(name) compile FAILED: \(error)"); exit(1)
    }
}
let mainLib = compile("MSL")
let q1gLib = compile("MSL_Q1G")
func pso(_ lib: MTLLibrary, _ name: String) -> MTLComputePipelineState {
    try! dev.makeComputePipelineState(function: lib.makeFunction(name: name)!)
}
let pipes = [
    "q1": pso(mainLib, "q1_matvec"), "q1h": pso(mainLib, "q1_matvec_h"),
    "q1g": pso(q1gLib, "q1g_matvec"), "q1gh": pso(q1gLib, "q1g_matvec_h"),
]
let queue = dev.makeCommandQueue()!
print("compiled MSL + MSL_Q1G on \(dev.name)")

var rng: UInt64 = 0x9E37_79B9_7F4A_7C15
func next() -> UInt64 { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng }

func run(_ kernel: String, _ w: [UInt8], _ off: Int, _ x: [Float], rows: Int, gpr: Int) -> [Float] {
    var wb = [UInt8](repeating: 0, count: off) + w
    while wb.count % 16 != 0 { wb.append(0) }
    let qb = dev.makeBuffer(bytes: wb, length: wb.count, options: .storageModeShared)!
    let xb = dev.makeBuffer(bytes: x, length: x.count * 4, options: .storageModeShared)!
    let yb = dev.makeBuffer(length: rows * 4, options: .storageModeShared)!
    let cmd = queue.makeCommandBuffer()!
    let enc = cmd.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipes[kernel]!)
    enc.setBuffer(qb, offset: off, index: 0)
    enc.setBuffer(xb, offset: 0, index: 1)
    enc.setBuffer(yb, offset: 0, index: 2)
    var g = UInt32(gpr), r = UInt32(rows)
    enc.setBytes(&g, length: 4, index: 3)
    enc.setBytes(&r, length: 4, index: 4)
    let sgs = 8
    enc.dispatchThreadgroups(MTLSize(width: (rows + sgs * 4 - 1) / (sgs * 4), height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: sgs * 32, height: 1, depth: 1))
    enc.endEncoding()
    cmd.commit()
    cmd.waitUntilCompleted()
    let p = yb.contents().bindMemory(to: Float.self, capacity: rows)
    return Array(UnsafeBufferPointer(start: p, count: rows))
}

var allOk = true
// Odd row counts (the kernels' 4-row tails), gpr = 4 and 12 (one and three
// groups per row; 3×384 has an odd group count), and Bonsai-1.7B's decode
// shapes up to the 151669-row head.
let shapes: [(Int, Int)] = [(1, 128), (3, 128), (5, 256), (3, 384), (33, 2048), (130, 128),
                            (7, 6144), (2048, 2048), (6144, 2048), (2048, 6144), (151_669, 2048)]
for seed in 0..<2 {
    for (rows, cols) in shapes {
        let gpr = cols / 32
        let tiles = rows * gpr
        let scOff = rows * cols / 8
        var q1 = [UInt8](repeating: 0, count: tiles * 6)
        var q1g = [UInt8](repeating: 0, count: scOff + rows * cols / 64)
        var scales = [Float](repeating: 0, count: tiles / 4)
        for g in 0..<(tiles / 4) {
            let s = Float16(0.004 + Float(next() % 3000) / 3000.0 * 0.03)
            scales[g] = Float(s)
            let h = s.bitPattern
            q1g[scOff + g * 2] = UInt8(h & 0xFF)
            q1g[scOff + g * 2 + 1] = UInt8(h >> 8)
            for t in 0..<4 {
                let ti = g * 4 + t
                let bits = UInt32(truncatingIfNeeded: next())
                q1[ti * 6] = UInt8(h & 0xFF)
                q1[ti * 6 + 1] = UInt8(h >> 8)
                for k in 0..<4 {
                    let b = UInt8((bits >> (8 * UInt32(k))) & 0xFF)
                    q1[ti * 6 + 2 + k] = b
                    q1g[ti * 4 + k] = b
                }
            }
        }
        var x = [Float](repeating: 0, count: cols)
        for i in 0..<cols {
            x[i] = (Float(next() % 20001) / 10000.0 - 1.0) * (i % 97 == 3 + seed ? 30 : 1)
        }
        let a = run("q1", q1, 128, x, rows: rows, gpr: gpr)
        let b = run("q1g", q1g, 192, x, rows: rows, gpr: gpr)
        let ah = run("q1h", q1, 64, x, rows: rows, gpr: gpr)
        let bh = run("q1gh", q1g, 256, x, rows: rows, gpr: gpr)
        let f32Diff = zip(a, b).filter { $0.bitPattern != $1.bitPattern }.count
        let halfDiff = zip(ah, bh).filter { $0.bitPattern != $1.bitPattern }.count
        var maxRel = 0.0
        for r in Set([0, rows / 2, rows - 1]) {
            var acc = 0.0
            for c in 0..<cols {
                let t = r * gpr + c / 32
                let bit = (q1[t * 6 + 2 + (c % 32) / 8] >> UInt8(c % 8)) & 1
                acc += Double(scales[t / 4]) * (bit == 1 ? 1 : -1) * Double(x[c])
            }
            maxRel = max(maxRel, abs(acc - Double(b[r])) / max(abs(acc), 1.0))
        }
        let ok = f32Diff == 0 && halfDiff == 0 && maxRel < 1e-4
        allOk = allOk && ok
        print(String(format: "seed %d %7d x %5d: f32 rows differing %d | half rows differing %d | q1g vs f64 max rel %.2e  %@",
                     seed, rows, cols, f32Diff, halfDiff, maxRel, ok ? "ok" : "FAIL"))
    }
}
print(allOk ? "METAL_Q1G_PARITY_OK" : "METAL_Q1G_PARITY_FAIL")
exit(allOk ? 0 : 1)
