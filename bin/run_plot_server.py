from flask import Flask, render_template, request
from bokeh.embed import components
import reportlib as rep
import argparse
import wgutils as wg

app = Flask(__name__, template_folder='.')
df = rep.generate_dummy_data()


@wg.start_stop
@app.route('/error_report')
def plot_error_report():
    """
    Plots an error report
    """
    fig = rep.generate_error_report(df, title='Rendered error report', save_to=None)
    script, div = components(fig)
    return render_template('template.html', script=script, div=div)


@wg.start_stop
@app.route('/delta_report')
def plot_delta_report():
    """
    Plots a delta report
    """
    on = request.args.get('vs', df.columns[0])
    fig = rep.generate_delta_report(df, [x for x in df.columns if x != on], on,
                                    title='Delta report vs {}'.format(on), save_to=None)
    script, div = components(fig)
    return render_template('template.html', script=script, div=div)


@wg.start_stop
@app.route('/scatter')
def plot_cluster_report():
    """
    Plots a delta report
    """
    x = request.args.get('x', df.columns[0])
    y = request.args.get('y', df.columns[1])
    c = request.args.get('c', df.columns[2])
    s = request.args.get('s', df.columns[3])
    fig = rep.plot_scatter_bokeh(df.copy(), x=x, y=y, c=c, s=s, title='Scatter plot')
    script, div = components(fig)
    return render_template('template.html', script=script, div=div)


@app.route('/')
def index():
    return "RepPlot server is online"


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Plot an execution run")
    parser.add_argument('-p', '--port', default=5000, type=str, help='Port used by the server')
    args = parser.parse_args()
    wg.tracelog('Server running in port {}'.format(args.port))
    app.run(debug=True, port=int(args.port))
