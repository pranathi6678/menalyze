# Menalyze MVP

## Workflow
1. Generate Dummy Data
```
cd backend/model
python generate_dummy_data.py
```
2. Train Model
```
python train.py
```
3. Evaluate Model
```
python evaluate.py
```
4. Run Backend API
```
cd ..
uvicorn app:app --reload
```
API will run at http://127.0.0.1:8000
