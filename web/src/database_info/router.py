# Builtins
import os
import sqlite3
# Installed
from fastapi import (
    APIRouter,
    Request,
    Depends,
    Query,
    Path,
    UploadFile,
    File
)
from fastapi.responses import (
    JSONResponse,
    FileResponse
)
# Local
from src.config import CONFIG
from src.core.vision_models import VisionModel
from src.core.database import (
    DatabaseManager,
    get_db_manager,
    DatabaseOperations
)
# Types
from typing import Annotated


database_info_bp = APIRouter(
    include_in_schema=True,
    tags=['Database-info']
)


@database_info_bp.get("/tableinfo")
async def tebleinfo(request: Request, table: Annotated[str, Query(description="...")]) -> JSONResponse:
    db = get_db_manager()
    dafuck = db.get_table_info(table)
    for column in dafuck:
        print(f"Column ID: {column['cid']}, Name: {column['name']}, Type: {column['type']}, Not Null: {column['notnull']}, Default: {column['dflt_value']}, Primary Key: {column['pk']}")
    return {"PATH": "image_path"}

@database_info_bp.get("/alltables")
async def alltables(request: Request) -> JSONResponse: # TODO: Change this for FileResponse
    db = get_db_manager()
    dafuck = db.get_all_tables()
    print(dafuck)
    return {"PATH": "image_path"}

@database_info_bp.get("/tex")
async def tex(request: Request) -> JSONResponse: # TODO: Change this for FileResponse
    db = get_db_manager()
    dafuck = db.table_exists("patients")
    print(dafuck)
    return {"PATH": "image_path"}

@database_info_bp.get("/rcount")
async def rcount(request: Request, table: Annotated[str, Query(description="...")], db: DatabaseManager = Depends(get_db_manager)) -> JSONResponse: # TODO: Change this for FileResponse
    # db = get_db_manager()
    dafuck = db.get_row_count(table)
    print(dafuck)
    from src.core.database import DatabaseOperations
    dbops = DatabaseOperations()
    data = db.execute_query(f"SELECT * FROM {table}", fetch="all")
    for row in data:
        if table == 'doctors':
            pwddec = db.decrypt_password(dict(row)['password'])
            print(pwddec)
            print(dbops.verify_password(dict(row)['id'], '1234asdf'))
            print(dbops.verify_password(dict(row)['id'], 'a pastar'))

        row_info = dict(row)
        if table =='pathologies':
            if row_info['pet_img'] is not None:
                row_info['pet_img'] = 'Hay content'
            if row_info['corrected_img'] is not None:
                row_info['corrected_img'] = 'Hay content'
        
        print(row_info)
    return {"PATH": "image_path"}
