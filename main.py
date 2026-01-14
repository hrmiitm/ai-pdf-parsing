import os
from google import genai
from pydantic import BaseModel, Field

class BoundingBoxField(BaseModel):
    bounding_box: list[int] = Field(..., description='The bounding box where the information was found [y_min, x_min, y_max, x_max]')
    page: int = Field(..., description='Page number where the information was found. Start counting with 1.')

class TotalAmountField(BoundingBoxField):
    value: float = Field(..., description='The total amount of the invoice.')

class RecipientField(BoundingBoxField):
    name: str = Field(..., description='The name of the recipient.')

class TaxAmountField(BoundingBoxField):
    value: float = Field(..., description='The total amount of the tax.')

class SenderField(BoundingBoxField):
    name: str = Field(..., description='The name of the sender.')

class AccountNumberField(BoundingBoxField):
    account_no: str = Field(..., description='The number of the account.')


class InvoiceModel(BaseModel):
    total: TotalAmountField
    recipient: RecipientField
    tax: TaxAmountField
    sender: SenderField
    account_no: AccountNumberField

def ai_analysis(pdf_file_loc: str, prompt: str, ai_model_name: str):
    client = genai.Client()
    pdf = client.files.upload(file=pdf_file_loc)

    response = client.models.generate_content(
        model=ai_model_name,
        contents=[pdf, prompt],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": InvoiceModel.model_json_schema(),
        },
    )

    invoice = InvoiceModel.model_validate_json(response.text)
    return invoice

def main():
    # Check Api Key available or not
    if not os.environ.get('GEMINI_API_KEY', None):
        raise RuntimeError("Gemini Api Key is NOT set") # using sys.exit(1) is bad idea, for library code/projectsz 
    else:
        print("1. Gemini Api Key is set")
    
    ai_model_name = "gemini-3-flash-preview"
    pdf_file_loc = 'invoice.pdf'

    prompt = """
    Extract the invoice recipient name and invoice total.
    Return only json that matches the provided schema.
    """
    print(f"2. AI Analysis on PDF started")
    invoice = ai_analysis(pdf_file_loc, prompt, ai_model_name)
    print(invoice)


if __name__ == "__main__":
    main()
