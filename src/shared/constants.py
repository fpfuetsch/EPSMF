import numpy as np

ignored_exercises = [
    "ExerciseThree",  # too few attempts
]

metric_curves = {
    "a_x": lambda x, a: a * x,
    "a_x_squared": lambda x, a: a * (x**2),
    "a_sqrt_x": lambda x, a: a * np.sqrt(x),
    "a_log_x": lambda x, a: a * np.log(x + 1),
    "a_p_log_x": lambda x, a: a + np.log(x + 1),
}

did_not_compile_test_name = "(did not compile)"
did_not_compile_replacement_name = "__NoCompilation__"

exercises_el = [
   {'title': 'ExerciseOne', 'n_tests': 3, 'mc_cabe_solution': 5, 'median_chars': 433, 'median_duration': 471.645}
]

type_mapping = {
    "exercise": "string",
    "attempt": "string",
    "difficulty": "string",
    "prior_xp": "string",
    "n_chars": int,
    "mc_cabe": int,
    "n_tests_green": int,
    "duration": float,
    "code_structure": int,
    "n_hints": int,
    "unused": int,
    "exercise_finished": bool,
    "author_confirmed": bool,
    "first_submission": "date",
    "group": "string",
    "tests_red": "string",
    "last_submission": "date",
    "last_progress": "date",
    "student": "string",
}


def reorder_record_data(df):
    order = [
        "exercise",
        "attempt",
        "student",
        "difficulty",
        "prior_xp",
        "group",
        "n_tests_green",
        "n_tests_red",
        "n_hints",
        "code_structure",
        "n_chars",
        "mc_cabe",
        "duration",
        "duration_effective",
        "first_submission",
        "last_submission",
        "last_progress",
        *sorted(df.columns[df.columns.str.startswith("metric_")].tolist()),
    ]
    return df[[field for field in order if field in df.columns]]

langusage_output_order = [
    "exercise",
    "mc_cabe_without_main",
    "mc_cabe_main",
    "IF",
    "SWITCH",
    "ASSIGNMENT",
    "BLOCK",
    "RETURN",
    "FOR",
    "FORENHANCED",
    "WHILE",
    "DOWHILE",
    "BREAK",
    "OBJECTCREATION",
    "DECLARATION",
    "DECLARATIONFRAGMENT",
    "INFIX",
    "PREFIX",
    "POSTFIX",
    "PREPOSTFIXINCDEC",
    "PREPOSTFIXINCDECFOR",
    "CAST",
    "METHODINVOCATION",
    "CONDITIONAL",
    "SIMPLENAME",
    "IMPORT",
    "BOOLEANLITERAL",
    "NULLLITERAL",
    "CHARACTERLITERAL",
    "NUMBERLITERAL",
    "STRINGLITERAL",
    "TYPELITERAL",
    "THIS",
    "PRIMITIVETYPE",
    "SIMPLETYPE",
    "ARRAYTYPE",
    "ARRAYACCESS",
    "ARRAYCREATION",
    "ARRAYINITIALIZER",
    "METHODDECLARATION",
    "FUNCTIONDECLARATION",
    "MAINDECLARATION",
    "FIELDDECLARATION",
    "STATICFIELDDECLARATION",
    "TYPEDECLARATION",
    "LONGCOMMENT",
    "INVOCATION",
    "FUNCTIONINVOCATION",
    "PASSEDARGUMENTS",
    "FIELDACCESS",
]

language_element_mapping = {
    "primitive": ["PRIMITIVETYPE", "INFIX", "PREFIX", "PREPOSTFIXINCDEC"],
    "control_flow": ["IF", "CONDITIONAL", "SWITCH", "FOR", "WHILE", "DOWHILE"],
    "arrays_functions": ["ARRAYTYPE", "ARRAYACCESS", "ARRAYCREATION", "ARRAYINITIALIZER", "FORENHANCED", "FUNCTIONDECLARATION", "STATICFIELDDECLARATION"],
    "classes": ["TYPEDECLARATION", "METHODDECLARATION", "FIELDDECLARATION", "THIS"],
}
