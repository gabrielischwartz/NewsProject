
from flask import Flask, render_template, request, make_response, Response
from news_actions import fetch_sources, summarize

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/generate', methods=['GET', 'POST'])
def generate():
    sources_mapping = fetch_sources()  # Fetch sources regardless of method

    selected_sources = request.cookies.get('selected_sources', '').split(',')  # Get selected sources from cookies

    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'update':
            selected_names = request.form.getlist('newsSource')  # Get selected names
            # Convert selected names to IDs
            selected_ids = [sources_mapping.get(name) for name in selected_names]
            # Store selected sources in a cookie
            resp = make_response(render_template('generate.html', sources=sources_mapping.keys(), selected_sources=selected_names))
            resp.set_cookie('selected_sources', ','.join(selected_names))
            return resp
        
        elif action == 'remove':
            source_to_remove = request.form.get('source')
            selected_sources = [s for s in selected_sources if s != source_to_remove]
            # Store updated selected sources in a cookie
            resp = make_response(render_template('generate.html', sources=sources_mapping.keys(), selected_sources=selected_sources))
            resp.set_cookie('selected_sources', ','.join(selected_sources))
            return resp
    

        elif action == 'generate':
            input_text = request.form.get('inputText', '')
            selected_names = request.cookies.get('selected_sources', '').split(',')  # Get selected sources from cookies
            # Convert selected names to IDs
            selected_ids = [sources_mapping.get(name) for name in selected_names]
            summary = summarize(input_text, selected_ids)
            # Store selected sources in a cookie
            resp = make_response(render_template('generate.html', sources=sources_mapping.keys(), summary=summary, selected_sources=selected_names))
            return resp
        


    # Render the page with the current selected sources
    selected_ids = [sources_mapping.get(name) for name in selected_sources]
    return render_template('generate.html', sources=sources_mapping.keys(), selected_sources=selected_sources)


if __name__ == '__main__':
    app.run(debug=True, port= 5001)

