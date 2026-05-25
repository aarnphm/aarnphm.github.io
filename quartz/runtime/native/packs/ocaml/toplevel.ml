open Js_of_ocaml
open Js_of_ocaml_toplevel

let stream name text =
  let callback = Js.Unsafe.get Js.Unsafe.global "quartzNativeOcamlStream" in
  match Js.Optdef.to_option callback with
  | None -> ()
  | Some fn ->
      ignore
        (Js.Unsafe.fun_call fn
           [|
             Js.Unsafe.inject (Js.string name);
             Js.Unsafe.inject (Js.string text);
           |])

let execute code =
  let code = Js.to_string code in
  let buffer = Buffer.create 256 in
  let formatter = Format.formatter_of_buffer buffer in
  JsooTop.execute true formatter code;
  Format.pp_print_flush formatter ();
  Js.string (Buffer.contents buffer)

let () =
  Sys_js.set_channel_flusher stdout (stream "stdout");
  Sys_js.set_channel_flusher stderr (stream "stderr");
  JsooTop.initialize ();
  Js.export "quartzNativeOcaml"
    (object%js
       val execute = execute
    end)
