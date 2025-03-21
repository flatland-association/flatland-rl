InteractiveAI OpenAPI Documents
===============================

[OpenAPI Generator CLI](https://pypi.org/project/openapi-generator-cli/) is used to generate API client libraries:

```
python -m pip install openapi-generator-cli
openapi-generator-cli generate -i flatland/integrations/interactiveai/openapi/openapi_context_api.json -g python --package-name flatland.integrations.interactiveai.context_api
openapi-generator-cli generate -i flatland/integrations/interactiveai/openapi/openapi_event_api.json -g python --package-name flatland.integrations.interactiveai.event_api
openapi-generator-cli generate -i flatland/integrations/interactiveai/openapi/openapi_historic_api.json -g python --package-name flatland.integrations.interactiveai.historic_api
```

[OpenAPI Documents](https://spec.openapis.org/oas/v3.1.1.html#openapi-document) are downloaded from
a [tweaked version of InteractiveAI](https://github.com/flatland-association/InteractiveAI/pull/1),
which exposes the data structures for Railway Use-Case in the document (the non-tweaked version only exposes a plain `dict` for the `data` attribute).
