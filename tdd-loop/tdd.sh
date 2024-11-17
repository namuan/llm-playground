#!/usr/bin/env bash
#
# A Bash script to run a TDD loop for building a Python module to pass tests.

set -euo pipefail

# How many times to loop.
ATTEMPTS=4

# The system prompt to use when creating the initial version.
INITIAL_PROMPT="
Write a Python module that will make these tests pass
and conforms to the passed conventions"

# The system prompt to use when creating subsequent versions.
RETRY_PROMPT="Tests are failing with this output. Try again."

function main {
    tests_file=$1
    app_file=$2
    test_output_file=$(mktemp)

    printf "Generating code to make these tests (in %s) pass\n\n" "$tests_file" >&2
    bat "$tests_file" >&2

    # Build initial version of application code.
    printf "\nGenerating initial version of %s\n\n" "$app_file" >&2
    files-to-prompt "$tests_file" conventions.txt | \
        llm prompt --system "$INITIAL_PROMPT" > "$app_file"

    for i in $(seq 2 $ATTEMPTS)
    do
        # Print output file.
        bat "$app_file" >&2

        # Pause for human inspection - otherwise everything flies past to quickly.
        echo >&2
        read -n 1 -s -r -p "Press any key to run tests..." >&2

        # Run tests and capture output.
        if pytest "$tests_file" > "$test_output_file"; then
            # Tests passed - we're done.
            echo "✅ " >&2
            exit 0
        else
            # Tests failed
            printf "❌\n\n" >&2
            bat "$test_output_file" >&2

            printf "\nGenerating v%s of %s\n\n" "$i" "$app_file" >&2
            files-to-prompt "$tests_file" conventions.txt "$test_output_file" | \
                llm prompt --continue --system "$RETRY_PROMPT" > "$app_file"
        fi
    done

    # If we get here, then no version passed the tests.
    echo "Failed to generate a passing version of $app_file after $ATTEMPTS attempts" >&2
    exit 1
}

main "$@"
