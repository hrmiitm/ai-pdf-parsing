import os
from google import genai
from pydantic import BaseModel, Field

import fitz

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

def anotate_pdf(pdf_path, anonted_pdf_path, items_to_draw):
    pdf_doc = fitz.open(pdf_path)
    
    for (label, box, page_no) in items_to_draw:
        if not box or box == [0,0,0,0] or page_no is None:
            continue
        
        page = pdf_doc[page_no - 1]
        (y0, x0, y1, x1) = box

        r = page.rect
        rect = fitz.Rect(
            (x0/1000) * r.width,
            (y0/1000) * r.height,
            (x1/1000) * r.width,
            (y1/1000) * r.height,
        )

        page.draw_rect(rect, color=(1,0,0), width=2)
        page.insert_text(
            (rect.x0, rect.y0 - 2),
            label,
            fontsize=6,
            color=(1,0,0)
        )
    pdf_doc.save(anonted_pdf_path)
    pdf_doc.close()

def main():
    # Check Api Key available or not
    if not os.environ.get('GEMINI_API_KEY', None):
        raise RuntimeError("Gemini Api Key is NOT set") # using sys.exit(1) is bad idea, for library code/projectsz 
    else:
        print("1. Gemini Api Key is set")
    
    ai_model_name = "gemini-3-flash-preview"
    pdf_file_loc = 'pdf/invoice_multipage.pdf'
    anotated_pdf_path = 'anotated_pdf/invoice_multipage.pdf'

    prompt = """
    Extract the invoice recipient name and invoice total.
    Return only json that matches the provided schema.
    """
    print(f"2. AI Analysis on PDF started")
    invoice = ai_analysis(pdf_file_loc, prompt, ai_model_name)
    # print(invoice)

    print(f"3. PDF anotation started")
    items_to_draw = [
        ("TOTAL", invoice.total.bounding_box, invoice.total.page),
        ("RECIPIENT", invoice.recipient.bounding_box, invoice.recipient.page),
        ("TAX", invoice.tax.bounding_box, invoice.tax.page),
        ("SENDER", invoice.sender.bounding_box, invoice.sender.page),
        ("ACCOUNT_NO", invoice.account_no.bounding_box, invoice.account_no.page)
    ]
    anotate_pdf(pdf_file_loc, anotated_pdf_path, items_to_draw)
    print(f"4. Anonted pdf saved in : {anotated_pdf_path}")
    print("--- Thank You ---")



if __name__ == "__main__":
    main()
