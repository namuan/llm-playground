from typing import Callable

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field


# Pydantic models for structured output
class LineChange(BaseModel):
    line_no: int = Field(description="Line number to change.")
    replacement: str = Field(description="New line to replace original.")


class TextRevision(BaseModel):
    chain_of_thought: str = Field(description="Reasoning for changes.")
    lines_changes: list[LineChange] = Field(description="List of line changes.")


# Validation function
def validate_contents(text: str, validator_func: Callable) -> str | None:
    try:
        validator_func(text)
        return None
    except Exception as e:
        return str(e)


# Get AI revision
def get_ai_revision(text: str, feedback: str) -> TextRevision:
    client = instructor.from_openai(OpenAI())
    numbered_text = "\n".join(
        f"{i:2d}: {line}" for i, line in enumerate(text.splitlines(), 1)
    )
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=TextRevision,
        messages=[
            {
                "role": "system",
                "content": "You are an expert editor. Fix text by replacing problematic lines based on feedback.",
            },
            {
                "role": "user",
                "content": f"Text:\n```\n{numbered_text}\n```\n\nFeedback:\n```\n{feedback}\n```",
            },
        ],
    )


# Apply changes to text
def apply_changes(text: str, changes: list[LineChange]) -> str:
    lines = text.splitlines()
    for change in changes:
        lines[change.line_no - 1] = change.replacement
    return "\n".join(lines)


# Main revision function
def revise_text(text: str, feedback: Callable | str) -> str:
    if not isinstance(feedback, str):
        feedback = validate_contents(text, feedback)
        if not feedback:
            return text
    revision = get_ai_revision(text, feedback)
    return apply_changes(text, revision.lines_changes)


# Example: Fix HoloViews code
code = """
import pandas as pd
import holoviews as hv
hv.extension('bokeh')

# Sample data
df = pd.DataFrame({
    'date': pd.date_range('2025-01-01', periods=5),
    'value': [1234.567, 2345.678, 3456.789, 4567.890, 5678.901],
})

# Create HoloViews Points
points = hv.Points(df, kdims=['date'], vdims=['value'])

# Apply hover formatting
points.opts(
    tools=['hover'],
    hover_tooltips=[
        ('Date', '@date{%F}'),
        ('Value', '@value{0,0.00}'),
    ],
    hover_formatters={
        '@date': 'datetime',
        '@value': 'numeral',
    },
    size=10,
    color='navy'
)
"""

# Fix code errors
fixed_code = revise_text(code, exec)
print("Fixed HoloViews code:\n", fixed_code)
