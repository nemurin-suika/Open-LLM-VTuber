#!/bin/bash

pushd frontend
npm run build:web
popd

# ahttp_client 1.0.5 버그 패치: pydantic 2.12+에서 by_alias/by_name=False로 model_validate
# 호출 시 에러 발생. None으로 바꿔서 pydantic이 모델 config 기준으로 동작하게 함.
PATCH_FILE=$(uv run python -c "import ahttp_client.extension.pydantic as m; import inspect; print(inspect.getfile(m))" 2>/dev/null)
if [ -n "$PATCH_FILE" ]; then
    sed -i '' 's/by_alias: Optional\[bool\] = False,/by_alias: Optional[bool] = None,/g' "$PATCH_FILE"
    sed -i '' 's/by_name: Optional\[bool\] = False,/by_name: Optional[bool] = None,/g' "$PATCH_FILE"
fi

uv run run_server.py