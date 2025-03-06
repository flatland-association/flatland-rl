```
python -m pip install openapi-generator-cli
openapi-generator-cli generate -i flatland/integrations/interactiveai/openapi/openapi_context_api.json -g python --package-name flatland.integrations.interactiveai.context_api
openapi-generator-cli generate -i flatland/integrations/interactiveai/openapi/openapi_event_api.json -g python --package-name flatland.integrations.interactiveai.event_api
openapi-generator-cli generate -i flatland/integrations/interactiveai/openapi/openapi_historic_api.json -g python --package-name flatland.integrations.interactiveai.historic_api
```
