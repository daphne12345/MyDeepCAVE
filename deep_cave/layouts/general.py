import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

from deep_cave.converter import converters
from deep_cave.server import app
from deep_cave.data_manager import dm
from deep_cave.run_manager import rm
from deep_cave.layouts.layout import Layout


class GeneralLayout(Layout):
    def _register_callbacks(self):
        outputs = [
            Output('general-working-directory-input', 'value'),
            Output('general-converter-select', 'value'),
            Output('general-runs-radiolist', 'options'),
            Output('general-runs-radiolist', 'value'),
            Output('general-alert', 'is_open'),
            Output('general-alert', 'children'),
        ]

        inputs = [
            Input('on-page-load', 'href'),
            Input('general-update-button', 'n_clicks'),
            State('general-working-directory-input', 'value'),
            State('general-converter-select', 'value')
        ]

        # Register updates from inputs
        @app.callback(outputs, inputs)
        def general_update(_, n_clicks, working_dir, converter_name):
            alert_open = False
            alert_message = None

            if isinstance(n_clicks, int) and n_clicks > 0:
                dm.clear()
                dm.set("working_dir", working_dir)
                dm.set("converter_name", converter_name)
                dm.set("run_id", "")

                alert_open = True
                alert_message = "Successfully updated meta data."
            
            return \
                GeneralLayout.get_working_dir(), \
                GeneralLayout.get_converter_name(), \
                GeneralLayout.get_run_options(), \
                GeneralLayout.get_run_id(), \
                alert_open, \
                alert_message

        input = Input('general-runs-radiolist', 'value')
        output = Output('general-runs-output', 'value')

        # Save the run ids internally
        # We have to inform the other plugins here as well
        @app.callback(output, input)
        def general_register_runs(run_id):
            if self.get_run_id() != run_id:
                working_dir = dm.get("working_dir")
                converter_name = dm.get("converter_name")

                # Clear cache
                dm.clear()

                # Set everything
                dm.set("working_dir", working_dir)
                dm.set("converter_name", converter_name)
                dm.set("run_id", run_id)

                return run_id
            
            raise PreventUpdate()

    @staticmethod
    def get_converter_options():
        return [{"label": adapter, "value": adapter} for adapter in converters.keys()]

    @staticmethod
    def get_run_options():
        return [{"label": run_name, "value": run_name} for run_name in rm.get_run_ids()]

    @staticmethod
    def get_run_id():
        run_id = dm.get("run_id")
        if run_id is None:
            return ""

        return run_id

    @staticmethod
    def get_working_dir():
        return dm.get("working_dir")

    @staticmethod
    def get_converter_name():
        return dm.get("converter_name")

    def _get_layout(self):
        return [
            html.H1('General'),

            dbc.Alert("", color="success", id="general-alert", is_open=False, dismissable=True),

            html.Div("Working Directory"),
            html.Div(html.I("Absolute path to your studies.")),
            dbc.Input(id="general-working-directory-input", placeholder="", type="text", 
                #value=GeneralLayout.get_working_dir()
            ),

            html.Div("Converter"),
            html.Div(html.I("Which optimizer was used to receive the data?")),
            dbc.Select(
                id="general-converter-select",
                options=GeneralLayout.get_converter_options(),
                #value=GeneralLayout.get_converter(),
            ),

            dbc.Button("Update", id="general-update-button", color="primary", className="mt-3"),
            html.Hr(),

            html.H2('Runs'),
            dbc.Input(id="general-runs-output", style="display: none;"),
            dbc.RadioItems(
                id="general-runs-radiolist",
                #options=GeneralLayout.get_run_options(),
                #value=GeneralLayout.get_run_ids()
            )
        ]