const std = @import("std");

const Input = struct {
    rows: [5][]const u8,
    row_count: usize,
    max_width: usize,
};

fn parseInput() Input {
    const data = @embedFile("d6.txt");
    var lines = std.mem.splitScalar(u8, data, '\n');
    var input: Input = .{ .rows = undefined, .row_count = 0, .max_width = 0 };

    while (lines.next()) |line| {
        if (input.row_count < 5 and line.len > 0) {
            input.rows[input.row_count] = line;
            input.row_count += 1;
        }
    }

    for (input.rows[0..input.row_count]) |row| {
        if (row.len > input.max_width) input.max_width = row.len;
    }

    return input;
}

fn isSeparator(input: *const Input, col: usize) bool {
    for (0..input.row_count) |r| {
        if (col < input.rows[r].len and input.rows[r][col] != ' ') {
            return false;
        }
    }
    return true;
}

fn getOperation(input: *const Input, start_col: usize, end_col: usize) u8 {
    if (input.row_count > 4) {
        const end = @min(end_col, input.rows[4].len);
        if (start_col < end) {
            for (input.rows[4][start_col..end]) |c| {
                if (c == '+' or c == '*') return c;
            }
        }
    }
    return '+';
}

// p1: each row has one number per problem (read horizontally)
fn p1() u128 {
    const input = parseInput();
    var total: u128 = 0;
    var col: usize = 0;

    while (col < input.max_width) {
        while (col < input.max_width and isSeparator(&input, col)) col += 1;
        if (col >= input.max_width) break;

        const start_col = col;
        while (col < input.max_width and !isSeparator(&input, col)) col += 1;

        var numbers: [4]?u64 = .{ null, null, null, null };
        for (0..4) |r| {
            if (r >= input.row_count) continue;
            const end = @min(col, input.rows[r].len);
            if (start_col >= end) continue;
            const segment = input.rows[r][start_col..end];
            const trimmed = std.mem.trim(u8, segment, " ");
            if (trimmed.len > 0) {
                numbers[r] = std.fmt.parseInt(u64, trimmed, 10) catch null;
            }
        }

        const op = getOperation(&input, start_col, col);
        var result: u128 = if (op == '*') 1 else 0;
        for (numbers) |maybe_n| {
            if (maybe_n) |n| {
                if (op == '*') result *= n else result += n;
            }
        }
        total += result;
    }

    return total;
}

// p2: each character column is a number (digits read top-to-bottom)
fn p2() u128 {
    const input = parseInput();
    var total: u128 = 0;
    var col: usize = 0;

    while (col < input.max_width) {
        while (col < input.max_width and isSeparator(&input, col)) col += 1;
        if (col >= input.max_width) break;

        const start_col = col;
        while (col < input.max_width and !isSeparator(&input, col)) col += 1;

        const op = getOperation(&input, start_col, col);
        var result: u128 = if (op == '*') 1 else 0;

        // each character column in [start_col, col) is a separate number
        for (start_col..col) |c| {
            var num: u64 = 0;
            var has_digit = false;

            // digits top-to-bottom = most significant first
            for (0..4) |r| {
                if (r >= input.row_count) continue;
                if (c < input.rows[r].len) {
                    const ch = input.rows[r][c];
                    if (ch >= '0' and ch <= '9') {
                        num = num * 10 + (ch - '0');
                        has_digit = true;
                    }
                }
            }

            if (has_digit) {
                if (op == '*') result *= num else result += num;
            }
        }

        total += result;
    }

    return total;
}

pub fn main() !void {
    std.debug.print("p1: {}\n", .{p1()});
    std.debug.print("p2: {}\n", .{p2()});
}
