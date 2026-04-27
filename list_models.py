import google.generativeai as ai; ai.configure(api_key='AIzaSyCLAYqEnLKVNVT04JfTo6Dqc8OKvbbv1fY');
models = [m.name for m in ai.list_models()]
with open('models_utf8.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(models))
