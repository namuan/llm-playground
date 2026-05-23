; === TYPE DEFINITIONS ===

(class_declaration
  name: (type_identifier) @name.definition.class) @definition.class

(protocol_declaration
  name: (type_identifier) @name.definition.interface) @definition.interface

(typealias_declaration
  name: (type_identifier) @name.definition.type) @definition.type

(associatedtype_declaration
  name: (type_identifier) @name.definition.type) @definition.type

; === ENUM CASE DEFINITIONS ===

(enum_entry
  name: (simple_identifier) @name.definition.enum) @definition.enum

; === METHOD DEFINITIONS inside class/struct/actor/extension bodies ===

(class_declaration
    (class_body
        [
            (function_declaration
                name: (simple_identifier) @name.definition.method
            )
            (subscript_declaration
                (parameter (simple_identifier) @name.definition.method)
            )
            (init_declaration "init" @name.definition.method)
            (deinit_declaration "deinit" @name.definition.method)
        ]
    )
) @definition.method

; === METHOD DEFINITIONS inside enum bodies (use enum_class_body) ===

(class_declaration
    (enum_class_body
        [
            (function_declaration
                name: (simple_identifier) @name.definition.method
            )
            (init_declaration "init" @name.definition.method)
        ]
    )
) @definition.method

; === METHOD DEFINITIONS inside protocol bodies ===

(protocol_declaration
    (protocol_body
        [
            (protocol_function_declaration
                name: (simple_identifier) @name.definition.method
            )
            (subscript_declaration
                (parameter (simple_identifier) @name.definition.method)
            )
            (init_declaration "init" @name.definition.method)
        ]
    )
) @definition.method

; === PROPERTY DEFINITIONS inside class/struct/actor/extension bodies ===

(class_declaration
    (class_body
        [
            (property_declaration
                (pattern (simple_identifier) @name.definition.property)
            )
        ]
    )
) @definition.property

; === PROPERTY DEFINITIONS inside enum bodies ===

(class_declaration
    (enum_class_body
        [
            (property_declaration
                (pattern (simple_identifier) @name.definition.property)
            )
        ]
    )
) @definition.property

; === PROPERTY DEFINITIONS (top-level) ===

(property_declaration
    (pattern (simple_identifier) @name.definition.property)
) @definition.property

; === FUNCTION DEFINITIONS (top-level) ===

(function_declaration
    name: (simple_identifier) @name.definition.function) @definition.function

; ===========================================================================
; REFERENCES
; ===========================================================================

; --- Call references: direct function/method calls like `foo()` ---

(call_expression
    (simple_identifier) @name.reference.call
) @reference.call

; --- Call references: method calls on objects like `obj.method()` ---

(call_expression
    (navigation_expression
        (navigation_suffix
            (simple_identifier) @name.reference.call))
) @reference.call

; --- Constructor calls: `MyType()` ---

(constructor_expression
    (user_type
        (type_identifier) @name.reference.call)
) @reference.call

; --- Macro invocations: `#someMacro(...)` ---

(macro_invocation
    (simple_identifier) @name.reference.call
) @reference.call

; --- Type references: usage of types in annotations, inheritance, generics ---

(user_type
    (type_identifier) @name.reference.type
) @reference.type

; --- Member access: `obj.property` (when not part of a call_expression) ---

(navigation_expression
    (navigation_suffix
        (simple_identifier) @name.reference.property)
) @reference.property

; --- Import references: `import Module` ---

(import_declaration
    (identifier
        (simple_identifier) @name.reference.module)
) @reference.module
