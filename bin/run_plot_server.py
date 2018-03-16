from flask import Flask, render_template, request
from bokeh.embed import components
from ecpclib import ECPCplotter
import os
import argparse
import wgutils as wg

data_dir = os.path.join(os.environ['PROJECT_ROOT'], 'output')
app = Flask(__name__, template_folder='.')
ecpc = ECPCplotter()
data_file = os.path.join(data_dir, os.listdir(data_dir)[-1], 'data.hdf5')


@app.route('/ecpc')
def plot_ecpc_performance():
    market = request.args.get('market', None)
    device = request.args.get('device', None)
    agency = request.args.get('agency', None)
    city = request.args.get('city', None)
    if market is None or device is None or agency is None or city is None:
        return ("ECPC plot must be specified for market, device, agency and city")
    else:
        fig = ecpc.plot_performance_ecpc(m=market, d=device, a=agency, c=city, save_to=None, engine='bokeh')
        script, div = components(fig)
        wg.tracelog("Printing contents of {}".format(data_file))
        return render_template('template.html', script=script, div=div)


@app.route('/performance')
def plot_performance():
    market = request.args.get('market', None)
    device = request.args.get('device', None)
    agency = request.args.get('agency', None)
    city = request.args.get('city', None)
    against = request.args.get('against', "revenue")
    fig = ecpc.plot_performance(m=market, d=device, a=agency, c=city, against=against, save_to=None, engine='bokeh')
    script, div = components(fig)
    wg.tracelog("Printing contents of {}".format(data_file))
    return render_template('template.html', script=script, div=div)


@app.route('/correction')
def plot_correction():
    fig = ecpc.plot_correction_revenue(save_to=None, engine='bokeh')
    script, div = components(fig)
    wg.tracelog("Printing contents of {}".format(data_file))
    return render_template('template.html', script=script, div=div)


@app.route('/correction_weight')
def plot_correction_weight():
    market = request.args.get('market', None)
    device = request.args.get('device', None)
    agency = request.args.get('agency', None)
    city = request.args.get('city', None)
    fig = ecpc.plot_correction_weight(m=market, d=device, a=agency, c=city, save_to=None, engine='bokeh')
    script, div = components(fig)
    wg.tracelog("Printing contents of {}".format(data_file))
    return render_template('template.html', script=script, div=div)


@app.route('/input')
def plot_volume_revenue():
    market = request.args.get('market', None)
    device = request.args.get('device', None)
    agency = request.args.get('agency', None)
    city = request.args.get('city', None)
    fig = ecpc.plot_volume_revenue(m=market, d=device, a=agency, c=city, save_to=None, engine='bokeh')
    script, div = components(fig)
    wg.tracelog("Printing contents of {}".format(data_file))
    return render_template('template.html', script=script, div=div)


@app.route('/gamma')
def plot_gamma():
    market = request.args.get('market', None)
    device = request.args.get('device', None)
    agency = request.args.get('agency', None)
    city = request.args.get('city', None)
    fig = ecpc.plot_gamma(m=market, d=device, a=agency, c=city, save_to=None, engine='bokeh')
    script, div = components(fig)
    wg.tracelog("Printing contents of {}".format(data_file))
    return render_template('template.html', script=script, div=div)


@app.route('/')
def index():
    return "ECPC plot server - welcome"


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Plot an execution run")
    parser.add_argument('-f', '--file', default=data_file, type=str, help='HDF5 file to display')
    parser.add_argument('-p', '--port', default=5000, type=str, help='Port used by the server')
    args = parser.parse_args()
    wg.tracelog("Loading contents of {}".format(args.file))
    ecpc.load_from_hdf5(args.file)
    app.run(debug=True, port=int(args.port))
