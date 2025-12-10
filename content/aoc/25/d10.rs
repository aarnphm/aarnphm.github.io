use regex::Regex;
use std::collections::{HashSet, VecDeque};
use std::fs;
use std::sync::LazyLock;

static PAT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[([.#]+)\]").unwrap());
static BTN_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\(([0-9,]+)\)").unwrap());
static TGT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\{([0-9,]+)\}").unwrap());

type Rational = (i64, i64); // (numerator, denominator)

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

fn rat_new(n: i64, d: i64) -> Rational {
    if d == 0 {
        panic!("division by zero");
    }
    let g = gcd(n, d);
    let sign = if d < 0 { -1 } else { 1 };
    (sign * n / g, sign * d / g)
}

fn rat_add(a: Rational, b: Rational) -> Rational {
    rat_new(a.0 * b.1 + b.0 * a.1, a.1 * b.1)
}

fn rat_sub(a: Rational, b: Rational) -> Rational {
    rat_new(a.0 * b.1 - b.0 * a.1, a.1 * b.1)
}

fn rat_mul(a: Rational, b: Rational) -> Rational {
    rat_new(a.0 * b.0, a.1 * b.1)
}

fn rat_div(a: Rational, b: Rational) -> Rational {
    rat_new(a.0 * b.1, a.1 * b.0)
}

fn rat_neg(a: Rational) -> Rational {
    (-a.0, a.1)
}
fn rat_zero() -> Rational {
    (0, 1)
}
fn rat_one() -> Rational {
    (1, 1)
}
fn rat_from(n: i64) -> Rational {
    (n, 1)
}
fn rat_is_zero(a: Rational) -> bool {
    a.0 == 0
}
fn rat_le(a: Rational, b: Rational) -> bool {
    a.0 * b.1 <= b.0 * a.1
}
fn rat_ge_zero(a: Rational) -> bool {
    a.0 >= 0
}
fn rat_is_int(a: Rational) -> bool {
    a.1 == 1 || a.0 % a.1 == 0
}
fn rat_to_int(a: Rational) -> i64 {
    a.0 / a.1
}

fn rat_floor(a: Rational) -> i64 {
    if a.0 >= 0 {
        a.0 / a.1
    } else {
        (a.0 - a.1 + 1) / a.1
    }
}

fn rat_ceil(a: Rational) -> i64 {
    if a.0 >= 0 {
        (a.0 + a.1 - 1) / a.1
    } else {
        a.0 / a.1
    }
}

fn parse_line(s: &str) -> (String, Vec<Vec<usize>>, Vec<i64>) {
    let pattern = PAT_RE.captures(s).unwrap()[1].to_string();

    let buttons: Vec<Vec<usize>> = BTN_RE
        .captures_iter(s)
        .map(|cap| {
            cap[1]
                .split(',')
                .map(|x| x.trim().parse().unwrap())
                .collect()
        })
        .collect();

    let targets: Vec<i64> = TGT_RE.captures(s).unwrap()[1]
        .split(',')
        .map(|x| x.trim().parse().unwrap())
        .collect();

    (pattern, buttons, targets)
}

// part 1: BFS over XOR states
fn pattern_to_mask(pat: &str) -> u64 {
    pat.chars()
        .enumerate()
        .fold(0, |acc, (i, c)| if c == '#' { acc | (1 << i) } else { acc })
}

fn button_to_mask(btn: &[usize]) -> u64 {
    btn.iter().fold(0, |acc, &i| acc ^ (1 << i))
}

fn min_press_lights(target: u64, masks: &[u64]) -> u32 {
    if target == 0 {
        return 0;
    }
    let mut seen = HashSet::new();
    seen.insert(0u64);
    let mut q = VecDeque::new();
    q.push_back((0u64, 0u32));

    while let Some((state, dist)) = q.pop_front() {
        for &m in masks {
            let nxt = state ^ m;
            if nxt == target {
                return dist + 1;
            }
            if !seen.contains(&nxt) {
                seen.insert(nxt);
                q.push_back((nxt, dist + 1));
            }
        }
    }
    panic!("unreachable");
}

// part 2: gaussian elimination
fn gauss_jordan(
    a: &[Vec<Rational>],
    b: &[Rational],
) -> (Vec<Vec<Rational>>, Vec<Rational>, Vec<usize>) {
    let rows = a.len();
    let cols = if rows > 0 { a[0].len() } else { 0 };
    let mut mat: Vec<Vec<Rational>> = a.to_vec();
    let mut rhs: Vec<Rational> = b.to_vec();
    let mut pivots = Vec::new();
    let mut r = 0;

    for c in 0..cols {
        let pivot_row = (r..rows).find(|&i| !rat_is_zero(mat[i][c]));
        let pr = match pivot_row {
            Some(p) => p,
            None => continue,
        };
        mat.swap(r, pr);
        rhs.swap(r, pr);

        let scale = mat[r][c];
        for j in 0..cols {
            mat[r][j] = rat_div(mat[r][j], scale);
        }
        rhs[r] = rat_div(rhs[r], scale);

        for i in 0..rows {
            if i != r && !rat_is_zero(mat[i][c]) {
                let factor = mat[i][c];
                for j in 0..cols {
                    mat[i][j] = rat_sub(mat[i][j], rat_mul(factor, mat[r][j]));
                }
                rhs[i] = rat_sub(rhs[i], rat_mul(factor, rhs[r]));
            }
        }
        pivots.push(c);
        r += 1;
        if r >= rows {
            break;
        }
    }
    (mat, rhs, pivots)
}

fn solve_square(mat: &[Vec<Rational>], rhs: &[Rational]) -> Option<Vec<Rational>> {
    let n = mat.len();
    let (mat2, rhs2, _) = gauss_jordan(mat, rhs);
    let mut sol = vec![rat_zero(); n];

    for i in (0..n).rev() {
        let lead = (0..n).find(|&j| !rat_is_zero(mat2[i][j]))?;
        let rest = (lead + 1..n).fold(rat_zero(), |acc, j| {
            rat_add(acc, rat_mul(mat2[i][j], sol[j]))
        });
        sol[lead] = rat_sub(rhs2[i], rest);
    }
    Some(sol)
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if n < k {
        return vec![];
    }
    let mut result = Vec::new();
    fn gen(
        start: usize,
        k: usize,
        n: usize,
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }
        for i in start..=(n - k + current.len()) {
            current.push(i);
            gen(i + 1, k, n, current, result);
            current.pop();
        }
    }
    gen(0, k, n, &mut Vec::new(), &mut result);
    result
}

fn bounds_for_free(
    constraints: &[(Vec<Rational>, Rational)],
    f: usize,
) -> (Vec<Rational>, Vec<Rational>) {
    let mut mins: Vec<Option<Rational>> = vec![None; f];
    let mut maxs = vec![rat_zero(); f];

    for subset in combinations(constraints.len(), f) {
        let mat: Vec<Vec<Rational>> = subset.iter().map(|&i| constraints[i].0.clone()).collect();
        let rhs: Vec<Rational> = subset.iter().map(|&i| constraints[i].1).collect();

        if let Some(sol) = solve_square(&mat, &rhs) {
            let feasible = constraints.iter().all(|(c, b)| {
                let dot = c
                    .iter()
                    .zip(&sol)
                    .fold(rat_zero(), |acc, (&ci, &si)| rat_add(acc, rat_mul(ci, si)));
                rat_le(dot, *b)
            });
            if feasible {
                for j in 0..f {
                    mins[j] = Some(match mins[j] {
                        None => sol[j],
                        Some(m) if rat_le(sol[j], m) => sol[j],
                        Some(m) => m,
                    });
                    if rat_le(maxs[j], sol[j]) {
                        maxs[j] = sol[j];
                    }
                }
            }
        }
    }
    (
        mins.into_iter().map(|m| m.unwrap_or(rat_zero())).collect(),
        maxs,
    )
}

fn min_press_jolts(buttons: &[Vec<usize>], targets: &[i64]) -> i64 {
    let m = targets.len();
    let n = buttons.len();

    let a: Vec<Vec<Rational>> = (0..m)
        .map(|i| {
            buttons
                .iter()
                .map(|btn| {
                    if btn.contains(&i) {
                        rat_one()
                    } else {
                        rat_zero()
                    }
                })
                .collect()
        })
        .collect();
    let b: Vec<Rational> = targets.iter().map(|&t| rat_from(t)).collect();

    let (a_red, b_red, pivots) = gauss_jordan(&a, &b);
    let free: Vec<usize> = (0..n).filter(|c| !pivots.contains(c)).collect();
    let f_count = free.len();

    if f_count == 0 {
        let pivot_vals: Vec<Rational> = (0..pivots.len()).map(|r| b_red[r]).collect();
        if pivot_vals.iter().all(|&v| rat_ge_zero(v) && rat_is_int(v)) {
            return pivot_vals.iter().map(|&v| rat_to_int(v)).sum();
        }
        panic!("no feasible solution");
    }

    let row_info: Vec<(Rational, Vec<Rational>)> = (0..pivots.len())
        .map(|r| (b_red[r], free.iter().map(|&f| a_red[r][f]).collect()))
        .collect();

    let mut constraints: Vec<(Vec<Rational>, Rational)> = row_info
        .iter()
        .map(|(b_val, coeff)| (coeff.clone(), *b_val))
        .collect();

    for j in 0..f_count {
        let neg: Vec<Rational> = (0..f_count)
            .map(|k| {
                if k == j {
                    rat_neg(rat_one())
                } else {
                    rat_zero()
                }
            })
            .collect();
        constraints.push((neg, rat_zero()));
    }

    let (mins, maxs) = bounds_for_free(&constraints, f_count);
    let bounds: Vec<(i64, i64)> = mins
        .iter()
        .zip(&maxs)
        .map(|(&mi, &mx)| (rat_ceil(mi).max(0), rat_floor(mx)))
        .collect();

    fn search(
        idx: usize,
        vals: &mut Vec<i64>,
        bounds: &[(i64, i64)],
        row_info: &[(Rational, Vec<Rational>)],
        f_count: usize,
    ) -> i64 {
        if idx == f_count {
            let pivot_vals: Vec<Rational> = row_info
                .iter()
                .map(|(b_val, coeff)| {
                    let sum = coeff
                        .iter()
                        .zip(vals.iter())
                        .fold(rat_zero(), |acc, (&c, &v)| {
                            rat_add(acc, rat_mul(c, rat_from(v)))
                        });
                    rat_sub(*b_val, sum)
                })
                .collect();
            if pivot_vals.iter().all(|&v| rat_ge_zero(v) && rat_is_int(v)) {
                return vals.iter().sum::<i64>()
                    + pivot_vals.iter().map(|&v| rat_to_int(v)).sum::<i64>();
            }
            return i64::MAX / 4;
        }
        let (lo, hi) = bounds[idx];
        let mut best = i64::MAX / 4;
        for v in lo..=hi {
            vals.push(v);
            best = best.min(search(idx + 1, vals, bounds, row_info, f_count));
            vals.pop();
        }
        best
    }

    search(0, &mut Vec::new(), &bounds, &row_info, f_count)
}

fn main() {
    let raw = fs::read_to_string("d10.txt").expect("cannot read d10.txt");
    let lines: Vec<&str> = raw.lines().filter(|l| !l.is_empty()).collect();

    let p1: u32 = lines
        .iter()
        .map(|ln| {
            let (pat, btns, _) = parse_line(ln);
            let target = pattern_to_mask(&pat);
            let masks: Vec<u64> = btns.iter().map(|b| button_to_mask(b)).collect();
            min_press_lights(target, &masks)
        })
        .sum();

    let p2: i64 = lines
        .iter()
        .map(|ln| {
            let (_, btns, targets) = parse_line(ln);
            min_press_jolts(&btns, &targets)
        })
        .sum();

    println!("p1: {}", p1);
    println!("p2: {}", p2);
}
