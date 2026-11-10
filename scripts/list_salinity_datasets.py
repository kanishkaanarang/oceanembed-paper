"""List all Copernicus Marine datasets whose product or dataset text mentions salinity: product id, dataset id, version."""
import copernicusmarine

cat = copernicusmarine.describe(contains=["salinity"], disable_progress_bar=True)
for prod in cat.products:
    for ds in prod.datasets:
        vers = [v.label for v in ds.versions]
        print(f"{prod.product_id} | {ds.dataset_id} | versions {vers}")
