use anyhow::{Context, Result};
use regex::Regex;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct MethodSignature {
    pub method_name: String,
    pub return_type: String,
    pub params: Vec<(String, String)>,
}

#[derive(Debug)]
pub struct TestData {
    pub inputs: Vec<String>,
    pub expected: String,
}

const CPP_KEYWORDS: &[&str] = &[
    "if", "else", "for", "while", "do", "switch", "case", "break", "continue", "return", "goto",
    "try", "catch", "throw", "new", "delete", "sizeof", "typeof", "static_cast", "dynamic_cast",
    "const_cast", "reinterpret_cast", "class", "struct", "union", "enum", "namespace", "using",
    "public", "private", "protected", "virtual", "override", "final", "static", "const",
    "volatile", "mutable", "inline", "extern", "register", "auto", "typedef", "template",
    "typename", "decltype", "nullptr", "true", "false", "this", "operator",
];

pub fn parse_cpp_signatures(source: &str) -> Result<Vec<MethodSignature>> {
    let re = Regex::new(r"(?m)^\s*([\w<>,\s\*&]+?)\s+(\w+)\s*\(\s*([^)]*)\s*\)\s*\{")?;

    let mut methods = vec![];
    for cap in re.captures_iter(source) {
        let return_type = cap[1].trim().to_string();
        let method_name = cap[2].to_string();
        let params_str = &cap[3];

        if method_name == "Solution"
            || method_name == "main"
            || CPP_KEYWORDS.contains(&method_name.as_str())
        {
            continue;
        }

        let params = parse_cpp_params(params_str)?;
        methods.push(MethodSignature {
            method_name,
            return_type,
            params,
        });
    }

    if methods.is_empty() {
        anyhow::bail!("no method found in solution")
    }
    Ok(methods)
}

fn parse_cpp_params(params_str: &str) -> Result<Vec<(String, String)>> {
    let params_str = params_str.trim();
    if params_str.is_empty() {
        return Ok(vec![]);
    }

    let mut params = vec![];
    for param in params_str.split(',') {
        let param = param.trim();
        if param.is_empty() {
            continue;
        }

        let parts: Vec<&str> = param.rsplitn(2, |c: char| c.is_whitespace() || c == '&' || c == '*').collect();
        if parts.len() < 2 {
            anyhow::bail!("invalid param: {}", param);
        }

        let name = parts[0].trim().trim_start_matches('&').trim_start_matches('*');
        let mut type_part = parts[1].trim().to_string();

        if param.contains('&') && !type_part.contains('&') {
            type_part.push('&');
        }
        if param.contains('*') && !type_part.contains('*') {
            type_part.push('*');
        }

        params.push((name.to_string(), type_part));
    }

    Ok(params)
}

pub fn parse_rs_signatures(source: &str) -> Result<Vec<MethodSignature>> {
    let re = Regex::new(
        r"(?m)^\s*(?:pub\s+)?fn\s+(\w+)\s*\(\s*([^)]*)\s*\)\s*(?:->\s*([\w<>,\s\[\]&']+))?\s*\{",
    )?;

    let mut methods = vec![];
    for cap in re.captures_iter(source) {
        let method_name = cap[1].to_string();
        if method_name == "main" || method_name == "new" {
            continue;
        }

        let params_str = &cap[2];
        let return_type = cap.get(3).map(|m| m.as_str().trim()).unwrap_or("()").to_string();

        let params = parse_rs_params(params_str)?;
        methods.push(MethodSignature {
            method_name,
            return_type,
            params,
        });
    }

    if methods.is_empty() {
        anyhow::bail!("no method found in solution")
    }
    Ok(methods)
}

fn parse_rs_params(params_str: &str) -> Result<Vec<(String, String)>> {
    let params_str = params_str.trim();
    if params_str.is_empty() {
        return Ok(vec![]);
    }

    let mut params = vec![];
    let mut depth = 0;
    let mut current = String::new();

    for c in params_str.chars() {
        match c {
            '<' | '(' | '[' => {
                depth += 1;
                current.push(c);
            }
            '>' | ')' | ']' => {
                depth -= 1;
                current.push(c);
            }
            ',' if depth == 0 => {
                if !current.trim().is_empty() {
                    if let Some((name, ty)) = parse_single_rs_param(&current)? {
                        params.push((name, ty));
                    }
                }
                current.clear();
            }
            _ => current.push(c),
        }
    }

    if !current.trim().is_empty() {
        if let Some((name, ty)) = parse_single_rs_param(&current)? {
            params.push((name, ty));
        }
    }

    Ok(params)
}

fn parse_single_rs_param(param: &str) -> Result<Option<(String, String)>> {
    let param = param.trim();
    if param.is_empty() || param == "&self" || param == "&mut self" || param == "self" {
        return Ok(None);
    }

    let parts: Vec<&str> = param.splitn(2, ':').collect();
    if parts.len() != 2 {
        anyhow::bail!("invalid rust param: {}", param);
    }

    let name = parts[0].trim().trim_start_matches("mut ").to_string();
    let ty = parts[1].trim().to_string();
    Ok(Some((name, ty)))
}

pub fn parse_test_file(path: &Path) -> Result<TestData> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;

    let parts: Vec<&str> = content.splitn(2, "---").collect();
    if parts.len() != 2 {
        anyhow::bail!("test file must have '---' separator: {}", path.display());
    }

    let inputs: Vec<String> = parts[0]
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(String::from)
        .collect();

    let expected = parts[1]
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");

    Ok(TestData { inputs, expected })
}

pub fn outputs_equal(actual: &str, expected: &str) -> bool {
    let norm_actual = normalize_output(actual);
    let norm_expected = normalize_output(expected);
    norm_actual == norm_expected
}

fn normalize_output(s: &str) -> String {
    s.chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>()
        .to_lowercase()
}
