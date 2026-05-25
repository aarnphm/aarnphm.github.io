package main

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"syscall/js"

	"github.com/traefik/yaegi/interp"
	"github.com/traefik/yaegi/stdlib"
)

type evalResult struct {
	OK    bool   `json:"ok"`
	Value string `json:"value,omitempty"`
	Error string `json:"error,omitempty"`
}

var session, sessionErr = newInterpreter()

func newInterpreter() (*interp.Interpreter, error) {
	i := interp.New(interp.Options{})
	if err := i.Use(stdlib.Symbols); err != nil {
		return nil, err
	}
	return i, nil
}

func encode(result evalResult) string {
	bytes, err := json.Marshal(result)
	if err != nil {
		return `{"ok":false,"error":"failed to encode Go evaluation result"}`
	}
	return string(bytes)
}

func hasPackage(source string) bool {
	for _, line := range strings.Split(source, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "//") {
			continue
		}
		return strings.HasPrefix(line, "package ")
	}
	return false
}

func wrappedProgram(source string) string {
	imports, body := splitLeadingImports(source)
	return "package main\n" + imports + "\nfunc main() {\n" + body + "\n}\n"
}

func splitLeadingImports(source string) (string, string) {
	var imports []string
	var body []string
	inImportBlock := false
	for _, line := range strings.Split(source, "\n") {
		trimmed := strings.TrimSpace(line)
		if inImportBlock {
			imports = append(imports, line)
			if trimmed == ")" {
				inImportBlock = false
			}
			continue
		}
		if len(body) == 0 && trimmed == "" {
			continue
		}
		if len(body) == 0 && strings.HasPrefix(trimmed, "import ") {
			imports = append(imports, line)
			if strings.HasSuffix(trimmed, "(") {
				inImportBlock = true
			}
			continue
		}
		body = append(body, line)
	}
	return strings.Join(imports, "\n"), strings.Join(body, "\n")
}

func valueString(value reflect.Value) string {
	if !value.IsValid() {
		return ""
	}
	if !value.CanInterface() {
		return value.String()
	}
	item := value.Interface()
	if item == nil {
		return ""
	}
	return fmt.Sprint(item)
}

func evalProgram(source string) evalResult {
	i, err := newInterpreter()
	if err != nil {
		return evalResult{OK: false, Error: err.Error()}
	}
	if _, err := i.Eval(source); err != nil {
		return evalResult{OK: false, Error: err.Error()}
	}
	return evalResult{OK: true}
}

func evalSource(source string) evalResult {
	if strings.TrimSpace(source) == "" {
		return evalResult{OK: true}
	}
	if hasPackage(source) {
		return evalProgram(source)
	}
	if sessionErr != nil {
		return evalResult{OK: false, Error: sessionErr.Error()}
	}
	value, err := session.Eval(source)
	if err == nil {
		return evalResult{OK: true, Value: valueString(value)}
	}
	return evalProgram(wrappedProgram(source))
}

func eval(_ js.Value, args []js.Value) interface{} {
	if len(args) == 0 {
		return encode(evalResult{OK: false, Error: "missing Go source"})
	}
	return encode(evalSource(args[0].String()))
}

func main() {
	js.Global().Set("quartzNativeGoEval", js.FuncOf(eval))
	select {}
}
