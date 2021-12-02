""" Shows how to use flask and matplotlib together.

Shows SVG, and png.
The SVG is easier to style with CSS, and hook JS events to in browser.

python3 -m venv venv
. ./venv/bin/activate
pip install flask matplotlib
python flask_matplotlib.py
"""
import io
import json
import random
import time
from flask import Flask, Response, request, render_template, redirect
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG

from matplotlib.figure import Figure
from wtforms import Form, StringField, TextAreaField, BooleanField
from flask_socketio import SocketIO, emit

from scripts.absorption_gaas import config_file, mat_file
from sbe.aux_functions import yaml_parser
from scripts.absorption_gaas import absorption
from sbe.semiconductors import GaAs, BandStructure3D, SemicondYAML

import eventlet
eventlet.monkey_patch()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
socketio.init_app(app, cors_allowed_origins="*")



global fig

class MyForm(Form):
    checkbox = 1
    config = TextAreaField('Config file', render_kw={"rows": 35, "cols": 70})
    mat_param = TextAreaField('Material parameters', render_kw={"rows": 24, "cols": 70})


@app.route("/")
def index():
    """ Returns html with the img tag for your plot.
    """

    form = MyForm()

    num_x_points = int(request.args.get("num_x_points", 50))
    # in a real app you probably want to use a flask template.
    form.config.data = config_file
    form.mat_param.data = mat_file

    # return render_template("index.html", num_x_points=num_x_points, form=form, status="Ready")
    from bokeh.embed import components
    from bokeh.resources import INLINE

    # grab the static resources
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    # render template
    script, div = components(fig)
    html = render_template(
        'index.html',
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        css_resources=css_resources,
        num_x_points = num_x_points, form = form, status = "Ready"
    )
    return html

@app.route("/save")
def save():
    """ Returns html with the img tag for your plot.
    """

    return redirect("/")


# @app.route("/matplot-as-image-<int:num_x_points>.png")
# def plot_png(num_x_points=50):
#     """ renders the plot on the fly.
#     """
#     # fig = Figure()
#     # axis = fig.add_subplot(1, 1, 1)
#     # x_points = range(num_x_points)
#     # axis.plot(x_points, [random.randint(1, 30) for x in x_points])
#
#     output = io.BytesIO()
#     FigureCanvasAgg(fig).print_png(output)
#     return Response(output.getvalue(), mimetype="image/png")

@app.route("/matplot-as-image-<int:num_x_points>.png")
def plot_png(num_x_points=50):
    """ renders the plot on the fly.
    """
    # fig = Figure()
    # axis = fig.add_subplot(1, 1, 1)
    # x_points = range(num_x_points)
    # axis.plot(x_points, [random.randint(1, 30) for x in x_points])

    from bokeh.embed import components
    from bokeh.resources import INLINE

    # grab the static resources
    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    # render template
    script, div = components(fig)
    html = render_template(
        'index.html',
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        css_resources=css_resources,
    )
    return html

# @app.route("/matplot-as-image-<int:num_x_points>.svg")
# def plot_svg(num_x_points=50):
#     """ renders the plot on the fly.
#     """
#     # fig = Figure()
#     # axis = fig.add_subplot(1, 1, 1)
#     # x_points = range(num_x_points)
#     # axis.plot(x_points, [random.randint(1, 30) for x in x_points])
#
#     output = io.BytesIO()
#     FigureCanvasSVG(fig).print_svg(output)
#     return Response(output.getvalue(), mimetype="image/svg+xml")


@socketio.on('long-running-event')
def handle_my_custom_event(input_json):
    time.sleep(5)

    params = yaml_parser(config_file)

    gaas = GaAs()
    bs = BandStructure3D(material=gaas)
    energy, ans = absorption(bs, **params)

    emit('processing-finished', json.dumps({'data': 'finished processing!'}))


def run_engine():

    params = yaml_parser(config_file)
    mat_params = yaml_parser(mat_file)

    mat = SemicondYAML(**mat_params)
    bs = BandStructure3D(material=mat)
    energy, ans, data = absorption(bs, **params)
    from apps.graphics import make_fig

    # import pickle
    #
    # with open('data.pkl', 'rb') as infile:
    #     data = pickle.load(infile)

    fig = make_fig(*data)

    return fig


if __name__ == "__main__":

    import json
    from bokeh.io.doc import curdoc
    from bokeh.document import Document

    global fig
    fig = run_engine()

    import webbrowser
    webbrowser.open("http://127.0.0.1:5000/")
    socketio.run(app)
