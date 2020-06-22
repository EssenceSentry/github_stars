<p align="center">
<img src="https://user-images.githubusercontent.com/609349/57987382-7e294500-7a35-11e9-9c6a-f73e0f1d3a1c.png" />
<br /><br />
<a href="https://badge.fury.io/py/dagster"><img src="https://badge.fury.io/py/dagster.svg"></>
<a href="https://coveralls.io/github/dagster-io/dagster?branch=master"><img src="https://coveralls.io/repos/github/dagster-io/dagster/badge.svg?branch=master"></a>
<a href="https://buildkite.com/dagster/dagster"><img src="https://badge.buildkite.com/888545beab829e41e5d7303db15525a2bc3b0f0e33a72759ac.svg?branch=master"></a>
<a href="https://docs.dagster.io/"></a>
</p>

# Dagster

Dagster is a system for building modern data applications.

**Elegant programming model:** Dagster provides a set of abstractions for building self-describing,
testable, and reliable data applications. It embraces the principles of functional data programming;
gradual, optional typing; and testability as a first-class value.

<p align="center">
<img src="https://user-images.githubusercontent.com/4531914/79161353-366b8480-7d90-11ea-83ce-c8a9522359d5.gif" />
</p>

**Flexible & incremental:** Dagster integrates with your existing tools and systems, and can invoke
any computation–whether it be Spark, Python, a Jupyter notebook, or SQL. It is also designed to work
with your existing systems like [Kubernetes](https://docs.dagster.io/docs/deploying/k8s).

<p align="center">
<img src="https://user-images.githubusercontent.com/4531914/79161365-3d929280-7d90-11ea-9216-c88cce41d3f1.gif" />
</p>

**Beautiful tools:** Dagster's development environment, dagit, is designed to facilitate local
development for data engineers, machine learning engineers, and data scientists. It also can be run
as a production service, to support operating, debugging, and maintaining large-scale production
data pipelines.

<p align="center">
<img src="https://user-images.githubusercontent.com/4531914/79161362-3bc8cf00-7d90-11ea-8974-17edbde3dc0d.gif" />
</p>

## Getting Started

### Installation

<p align="center">
<code>pip install dagster dagit</code>
</p>

This installs two modules:

- **Dagster**: the core programming model and abstraction stack; stateless, single-node,
  single-process and multi-process execution engines; and a CLI tool for driving those engines.
- **Dagit**: the UI for developing and operating Dagster pipelines, including a DAG browser, a
  type-aware config editor, and a live execution interface.

### Hello dagster 👋

**`hello_dagster.py`**

```python
from dagster import execute_pipeline, pipeline, solid


@solid
def get_name(_):
    return 'dagster'


@solid
def hello(context, name: str):
    context.log.info('Hello, {name}!'.format(name=name))


@pipeline
def hello_pipeline():
    hello(get_name())
```

Save the code above in a file named `hello_dagster.py`. You can execute the pipeline using any one
of the following methods:

**(1) Dagster Python API**

```python
if __name__ == "__main__":
    execute_pipeline(hello_pipeline)   # Hello, dagster!
```

**(2) Dagster CLI**

```bash
$ dagster pipeline execute -f hello_dagster.py
```

**(3) Dagit web UI**

```bash
$ dagit -f hello_dagster.py
```

## Learn

Next, jump right into our [tutorial](https://docs.dagster.io/docs/tutorial/), or read our [complete
documentation](https://docs.dagster.io). If you're actively using Dagster or have questions on
getting started, we'd love to hear from you:

<br />
<p align="center">
<a href="https://join.slack.com/t/dagster/shared_invite/enQtNjEyNjkzNTA2OTkzLTI0MzdlNjU0ODVhZjQyOTMyMGM1ZDUwZDQ1YjJmYjI3YzExZGViMDI1ZDlkNTY5OThmYWVlOWM1MWVjN2I3NjU"><img src="https://user-images.githubusercontent.com/609349/63558739-f60a7e00-c502-11e9-8434-c8a95b03ce62.png" width=160px; /></a>
</p>

## Contributing

For details on contributing or running the project for development, check out our [contributing
guide](https://docs.dagster.io/docs/community/contributing/). <br />

## Integrations

Dagster works with the tools and systems that you're already using with your data, including:

<table>
	<thead>
		<tr style="background-color: #ddd" align="center">
			<td colspan=2><b>Integration</b></td>
			<td><b>Dagster Library</b></td>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td align="center" style="border-right: 0px"><img style="vertical-align:middle"  src="https://user-images.githubusercontent.com/609349/57987547-a7e36b80-7a37-11e9-95ae-4c4de2618e87.png"></td>
			<td style="border-left: 0px"> <b>Apache Airflow</b></td>
			<td><a href="https://docs.dagster.io/docs/apidocs/libraries/dagster_airflow" />dagster-airflow</a><br />Allows Dagster pipelines to be scheduled and executed, either containerized or uncontainerized, as <a href="https://github.com/apache/airflow">Apache Airflow DAGs</a>.</td>
		</tr>
		<tr>
			<td align="center" style="border-right: 0px"><img style="vertical-align:middle"  src="https://user-images.githubusercontent.com/609349/57987976-5ccc5700-7a3d-11e9-9fa5-1a51299b1ccb.png"></td>
			<td style="border-left: 0px"> <b>Apache Spark</b></td>
			<td><a href="https://docs.dagster.io/docs/apidocs/libraries/dagster_spark" />dagster-spark</a> &middot; <a href="https://docs.dagster.io/docs/apidocs/libraries/dagster_pyspark" />dagster-pyspark</a>
			<br />Libraries for interacting with Apache Spark and PySpark.
			</td>
		</tr>
		<tr>
			<td align="center" style="border-right: 0px"><img style="vertical-align:middle"  src="https://user-images.githubusercontent.com/609349/58348728-48f66b80-7e16-11e9-9e9f-1a0fea9a49b4.png"></td>
			<td style="border-left: 0px"> <b>Dask</b></td>
			<td><a href="https://docs.dagster.io/docs/apidocs/libraries/dagster_dask" />dagster-dask</a>
			<br />Provides a Dagster integration with Dask / Dask.Distributed.
			</td>
		</tr>
		<tr>
			<td align="center" style="border-right: 0px"><img style="vertical-align:middle" src="https://user-images.githubusercontent.com/609349/58349731-f36f8e00-7e18-11e9-8a2e-86e086caab66.png"></td>
			<td style="border-left: 0px"> <b>Datadog</b></td>
			<td><a href="https://docs.dagster.io/docs/apidocs/libraries/dagster_datadog" />dagster-datadog</a>
			<br />Provides a Dagster resource for publishing metrics to Datadog.
			</td>
		</tr>
		<tr>
			<td align="center" style="border-right: 0px"><img style="vertical-align:middle" src="https://user-images.githubusercontent.com/609349/57987809-bf245800-7a3b-11e9-8905-494ed99d0852.png" />
			&nbsp;/&nbsp; <img style="vertical-align:middle" src="https://user-images.githubusercontent.com/609349/57987827-fa268b80-7a3b-11e9-8a18-b675d76c19aa.png">
			</td>
			<td style="border-left: 0px"> <b>Jupyter / Papermill</b></td>
			<td><a href="https://docs.dagster.io/docs/apidocs/libraries/dagstermill" />dagstermill</a><br />Built on the <a href="https://github.com/nteract/papermill">papermill library</a>, dagstermill is meant for integrating productionized Jupyter notebooks into dagster pipelines.</td>
		</tr>
		<tr>
			<td align="center" style="border-right: 0px"><img style="vertical-align:middle"  src="https://user-images.githubusercontent.com/609349/57988016-f431aa00-7a3d-11e9-8cb6-1309d4246b27.png"></td>
			<td style="border-left: 0px"> <b>PagerDuty</b></td>
			<td><a href="https://docs.dagster.io/docs/apidocs/libraries/dagster_pagerduty" />dagster-pagerduty</a>
			<br />A library for creating PagerDuty alerts from Dagster workflows.
			</td>
		</tr>
		<tr>
			<td align="center" style="border-right: 0px"><img style="vertical-align:middle" src="https://user-images.githubusercontent.com/609349/58349397-fcac2b00-7e17-11e9-900c-9ab8cf7cb64a.png"></td>
			<td style="border-left: 0px"> <b>Snowflake</b></td>
			<td><a href="https://docs.dagster.io/docs/apidocs/libraries/dagster_snowflake" />dagster-snowflake</a>
			<br />A library for interacting with the Snowflake Data Warehouse.
			</td>
		</tr>
		<tr style="background-color: #ddd">
			<td colspan=2 align="center"><b>Cloud Providers</b></td>
			<td><b></b></td>
		</tr>
		<tr>
			<td align="center" style="border-right: 0px"><img style="vertical-align:middle" src="https://user-images.githubusercontent.com/609349/57987557-c2b5e000-7a37-11e9-9310-c274481a4682.png"> </td>
			<td style="border-left: 0px"><b>AWS</b></td>
			<td><a href="https://docs.dagster.io/docs/apidocs/libraries/dagster_aws" />dagster-aws</a>
			<br />A library for interacting with Amazon Web Services. Provides integrations with Cloudwatch, S3, EMR, and Redshift.
			</td>
		</tr>
		<tr>
			<td align="center" style="border-right: 0px"><img style="vertical-align:middle" src="https://user-images.githubusercontent.com/609349/84176312-0bbb4680-aa36-11ea-9580-a70758b12161.png"> </td>
			<td style="border-left: 0px"><b>Azure</b></td>
			<td><a href="https://docs.dagster.io/docs/apidocs/libraries/dagster_azure" />dagster-azure</a>
			<br />A library for interacting with Microsoft Azure.
			</td>
		</tr>
		<tr>
			<td align="center" style="border-right: 0px"><img style="vertical-align:middle" src="https://user-images.githubusercontent.com/609349/57987566-f98bf600-7a37-11e9-81fa-b8ca1ea6cc1e.png"> </td>
			<td style="border-left: 0px"><b>GCP</b></td>
			<td><a href="https://docs.dagster.io/docs/apidocs/libraries/dagster_gcp" />dagster-gcp</a>
			<br />A library for interacting with Google Cloud Platform. Provides integrations with GCS, BigQuery, and Cloud Dataproc.
			</td>
		</tr>
	</tbody>
</table>

This list is growing as we are actively building more integrations, and we welcome contributions!
