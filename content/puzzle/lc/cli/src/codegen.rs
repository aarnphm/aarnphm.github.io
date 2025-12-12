use crate::parser::{MethodSignature, TestData};
use anyhow::Result;
use std::path::Path;

const CPP_PRELUDE: &str = include_str!("../../prelude.h");

pub fn generate_cpp_harness(
    solution_path: &Path,
    sig: &MethodSignature,
    test: &TestData,
) -> Result<String> {
    let solution_include = solution_path.to_str().unwrap();

    let input_parsing = generate_cpp_input_parsing(sig, test)?;
    let call_args = sig
        .params
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>()
        .join(", ");

    let output_printing = generate_cpp_output_printing(&sig.return_type);

    Ok(format!(
        r#"{prelude}
#include "{solution_include}"

{input_parsing}

int main() {{
    Solution sol;
    auto t0 = chrono::high_resolution_clock::now();
    auto result = sol.{method}({call_args});
    auto t1 = chrono::high_resolution_clock::now();
    auto us = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();
    cerr << "TIME_US:" << us << endl;
    {output_printing}
    return 0;
}}
"#,
        prelude = CPP_PRELUDE.trim(),
        solution_include = solution_include,
        input_parsing = input_parsing,
        method = sig.method_name,
        call_args = call_args,
        output_printing = output_printing,
    ))
}

fn generate_cpp_input_parsing(sig: &MethodSignature, test: &TestData) -> Result<String> {
    let mut lines = vec![];

    for (i, (name, ty)) in sig.params.iter().enumerate() {
        let input = test.inputs.get(i).map(|s| s.as_str()).unwrap_or("");
        let decl = generate_cpp_var_decl(name, ty, input)?;
        lines.push(decl);
    }

    Ok(lines.join("\n"))
}

fn generate_cpp_var_decl(name: &str, ty: &str, value: &str) -> Result<String> {
    let base_type = ty.trim_end_matches('&').trim_end_matches('*').trim();

    match base_type {
        "int" => Ok(format!("int {} = {};", name, value)),
        "long long" => Ok(format!("long long {} = {};", name, value)),
        "double" => Ok(format!("double {} = {};", name, value)),
        "bool" => {
            let v = if value == "true" || value == "1" { "true" } else { "false" };
            Ok(format!("bool {} = {};", name, v))
        }
        "string" => {
            let s = value.trim_matches('"');
            Ok(format!("string {} = \"{}\";", name, s))
        }
        t if t.starts_with("vector<vector<") => {
            let inner = extract_inner_type(t, 2);
            Ok(format!(
                "vector<vector<{}>> {} = {};",
                inner,
                name,
                convert_2d_array(value, &inner)
            ))
        }
        t if t.starts_with("vector<") => {
            let inner = extract_inner_type(t, 1);
            Ok(format!(
                "vector<{}> {} = {};",
                inner,
                name,
                convert_1d_array(value, &inner)
            ))
        }
        _ => Ok(format!("auto {} = {}; // unsupported type: {}", name, value, ty)),
    }
}

fn extract_inner_type(ty: &str, depth: usize) -> String {
    let mut s = ty;
    for _ in 0..depth {
        if let Some(start) = s.find('<') {
            if let Some(end) = s.rfind('>') {
                s = &s[start + 1..end];
            }
        }
    }
    s.trim().to_string()
}

fn convert_1d_array(value: &str, inner: &str) -> String {
    let v = value.trim();
    if v == "[]" {
        return "{}".to_string();
    }

    let inner_str = v.trim_start_matches('[').trim_end_matches(']');

    if inner == "string" {
        let items: Vec<&str> = parse_string_array(inner_str);
        let formatted: Vec<String> = items
            .iter()
            .map(|s| format!("\"{}\"", s.trim().trim_matches('"')))
            .collect();
        format!("{{{}}}", formatted.join(", "))
    } else {
        format!("{{{}}}", inner_str)
    }
}

fn convert_2d_array(value: &str, inner: &str) -> String {
    let v = value.trim();
    if v == "[]" || v == "[[]]" {
        return "{}".to_string();
    }

    let inner_str = v.trim_start_matches('[').trim_end_matches(']');
    let rows = parse_nested_arrays(inner_str);

    let formatted: Vec<String> = rows
        .iter()
        .map(|row| convert_1d_array(row, inner))
        .collect();

    format!("{{{}}}", formatted.join(", "))
}

fn parse_string_array(s: &str) -> Vec<&str> {
    let mut result = vec![];
    let mut in_string = false;
    let mut start = 0;

    for (i, c) in s.char_indices() {
        match c {
            '"' => in_string = !in_string,
            ',' if !in_string => {
                let item = s[start..i].trim();
                if !item.is_empty() {
                    result.push(item);
                }
                start = i + 1;
            }
            _ => {}
        }
    }

    let last = s[start..].trim();
    if !last.is_empty() {
        result.push(last);
    }

    result
}

fn parse_nested_arrays(s: &str) -> Vec<String> {
    let mut result = vec![];
    let mut depth = 0;
    let mut start = 0;

    for (i, c) in s.char_indices() {
        match c {
            '[' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            ']' => {
                depth -= 1;
                if depth == 0 {
                    result.push(s[start..=i].to_string());
                }
            }
            _ => {}
        }
    }

    result
}

fn generate_cpp_output_printing(return_type: &str) -> String {
    let base_type = return_type.trim();

    match base_type {
        "int" | "long long" | "double" => "cout << result << endl;".to_string(),
        "bool" => "cout << (result ? \"true\" : \"false\") << endl;".to_string(),
        "string" => "cout << \"\\\"\" << result << \"\\\"\" << endl;".to_string(),
        t if t.starts_with("vector<vector<") => {
            r#"cout << "[";
    for (size_t i = 0; i < result.size(); i++) {
        if (i > 0) cout << ",";
        cout << "[";
        for (size_t j = 0; j < result[i].size(); j++) {
            if (j > 0) cout << ",";
            cout << result[i][j];
        }
        cout << "]";
    }
    cout << "]" << endl;"#.to_string()
        }
        t if t.starts_with("vector<") => {
            let inner = extract_inner_type(t, 1);
            if inner == "string" {
                r#"cout << "[";
    for (size_t i = 0; i < result.size(); i++) {
        if (i > 0) cout << ",";
        cout << "\"" << result[i] << "\"";
    }
    cout << "]" << endl;"#.to_string()
            } else {
                r#"cout << "[";
    for (size_t i = 0; i < result.size(); i++) {
        if (i > 0) cout << ",";
        cout << result[i];
    }
    cout << "]" << endl;"#.to_string()
            }
        }
        _ => format!("cout << result << endl; // unknown type: {}", return_type),
    }
}

pub fn generate_rs_harness(
    solution_path: &Path,
    sig: &MethodSignature,
    test: &TestData,
) -> Result<String> {
    let solution_content = std::fs::read_to_string(solution_path)?;

    let input_parsing = generate_rs_input_parsing(sig, test)?;
    let call_args = sig
        .params
        .iter()
        .map(|(name, ty)| {
            if ty.starts_with("&mut") {
                format!("&mut {}", name)
            } else if ty.starts_with('&') {
                format!("&{}", name)
            } else {
                name.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(", ");

    let output_printing = generate_rs_output_printing(&sig.return_type);

    Ok(format!(
        r#"{solution_content}

fn main() {{
{input_parsing}
    let sol = Solution;
    let result = sol.{method}({call_args});
    {output_printing}
}}
"#,
        solution_content = solution_content,
        input_parsing = input_parsing,
        method = sig.method_name,
        call_args = call_args,
        output_printing = output_printing,
    ))
}

fn generate_rs_input_parsing(sig: &MethodSignature, test: &TestData) -> Result<String> {
    let mut lines = vec![];

    for (i, (name, ty)) in sig.params.iter().enumerate() {
        let input = test.inputs.get(i).map(|s| s.as_str()).unwrap_or("");
        let decl = generate_rs_var_decl(name, ty, input)?;
        lines.push(format!("    {}", decl));
    }

    Ok(lines.join("\n"))
}

fn generate_rs_var_decl(name: &str, ty: &str, value: &str) -> Result<String> {
    let base_type = ty
        .trim_start_matches("&mut ")
        .trim_start_matches('&')
        .trim();

    match base_type {
        "i32" | "i64" | "usize" | "f64" => Ok(format!("let mut {} = {};", name, value)),
        "bool" => {
            let v = if value == "true" || value == "1" { "true" } else { "false" };
            Ok(format!("let {} = {};", name, v))
        }
        "String" => {
            let s = value.trim_matches('"');
            Ok(format!("let {} = \"{}\".to_string();", name, s))
        }
        t if t.starts_with("Vec<Vec<") => {
            Ok(format!("let mut {} = vec!{};", name, value.trim().to_string()))
        }
        t if t.starts_with("Vec<") => {
            Ok(format!("let mut {} = vec!{};", name, value.trim().to_string()))
        }
        _ => Ok(format!("let mut {} = {}; // type: {}", name, value, ty)),
    }
}

fn generate_rs_output_printing(return_type: &str) -> String {
    let base_type = return_type.trim();

    match base_type {
        "i32" | "i64" | "usize" | "f64" => "println!(\"{}\", result);".to_string(),
        "bool" => "println!(\"{}\", result);".to_string(),
        "String" => "println!(\"\\\"{}\\\"\", result);".to_string(),
        t if t.starts_with("Vec<") => "println!(\"{:?}\", result);".to_string(),
        _ => format!("println!(\"{{:?}}\", result); // type: {}", return_type),
    }
}
