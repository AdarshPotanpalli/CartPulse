
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from typing import Optional

# Importing constants and pipeline modules from the project
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import ProductData, ProductDataClassifier
from src.pipline.training_pipeline import TrainPipeline

# Initialize FastAPI application
app = FastAPI()

# Mount the 'static' directory for serving static files
app.mount("/static", StaticFiles(directory="static"), name="static")
# Set up Jinja2 template engine
templates = Jinja2Templates(directory="templates")


# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:
    """
    Class to handle and process incoming product/customer form data.
    """
    def __init__(self, request: Request):

        self.request: Request = request
        self.ProductCategory: Optional[str] = None
        self.ProductBrand: Optional[str] = None
        self.ProductPrice: Optional[float] = None
        self.CustomerAge: Optional[int] = None
        self.CustomerGender: Optional[int] = None
        self.PurchaseFrequency: Optional[int] = None
        self.CustomerSatisfaction: Optional[int] = None

    async def get_product_data(self):
        """
        Retrieve and assign form data to class attributes.
        """
        form = await self.request.form()
        self.ProductCategory = form.get("ProductCategory")
        self.ProductBrand = form.get("ProductBrand")
        self.ProductPrice = form.get("ProductPrice")
        self.CustomerAge = form.get("CustomerAge")
        self.CustomerGender = form.get("CustomerGender")
        self.PurchaseFrequency = form.get("PurchaseFrequency")
        self.CustomerSatisfaction = form.get("CustomerSatisfaction")

# -------------------------------------------------------------------
# Home page
# -------------------------------------------------------------------
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="productdata.html",
        context={}
    )


# -------------------------------------------------------------------
# Training endpoint
# -------------------------------------------------------------------
@app.get("/train")
async def trainRouteClient(request: Request):
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

        return templates.TemplateResponse(
            request=request,
            name="productdata.html",
            context={
                "training_status": "Training successful!!!"
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="productdata.html",
            context={
                "training_status": f"Error Occurred! {e}"
            }
        )

# -------------------------------------------------------------------
# Prediction endpoint
# -------------------------------------------------------------------
@app.post("/")
async def predictRouteClient(request: Request):
    """
    Receive product/customer data and make a PurchaseIntent prediction.
    """
    try:
        # -----------------------------------------------------------
        # 1. Read form data
        # -----------------------------------------------------------
        form = DataForm(request)
        await form.get_product_data()
        # -----------------------------------------------------------
        # 2. Create ProductData object
        # -----------------------------------------------------------
        product_data = ProductData(
            ProductCategory=form.ProductCategory,
            ProductBrand=form.ProductBrand,
            ProductPrice=float(form.ProductPrice),
            CustomerAge=int(form.CustomerAge),
            CustomerGender=int(form.CustomerGender),
            PurchaseFrequency=int(form.PurchaseFrequency),
            CustomerSatisfaction=int(form.CustomerSatisfaction)
        )

        # -----------------------------------------------------------
        # 3. Convert input into DataFrame
        # -----------------------------------------------------------
        product_df = product_data.get_product_input_data_frame()
        
        # -----------------------------------------------------------
        # 4. Initialize prediction pipeline
        # -----------------------------------------------------------
        model_predictor = ProductDataClassifier()


        # -----------------------------------------------------------
        # 5. Make prediction
        # -----------------------------------------------------------
        value = model_predictor.predict(
            dataframe=product_df
        )[0]

        # -----------------------------------------------------------
        # 6. Interpret prediction
        # -----------------------------------------------------------
        status = (
            "YES"
            if value == 1
            else "NO"
        )

        # -----------------------------------------------------------
        # 7. Render result
        # -----------------------------------------------------------
        return templates.TemplateResponse(
            request=request,
            name="productdata.html",
            context={
                "context": status
            }
        )


    except Exception as e:
        return {
            "status": False,
            "error": f"{e}"
        }


# -------------------------------------------------------------------
# Run application
# -------------------------------------------------------------------
if __name__ == "__main__":
    app_run(
        app,
        host=APP_HOST,
        port=APP_PORT
    )

