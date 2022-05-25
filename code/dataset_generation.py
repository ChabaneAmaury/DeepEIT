import datetime
from ast import literal_eval
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from os import makedirs
from os.path import exists, join
from random import randint, uniform

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.image import imsave
from pyeit import mesh
from pyeit.eit import bp, jac
from pyeit.eit.fem import Forward
from pyeit.eit.interp2d import sim2pts
from pyeit.eit.utils import eit_scan_lines
from pyeit.mesh.shape import circle
from tqdm import tqdm

import mongodbAPI


def back_projection(n_el, anomaly):
    """ 0. build mesh """
    # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax , Default :fd=circle
    mesh_obj, el_pos = mesh.create(n_el=n_el,
                                   fd=circle,
                                   h0=0.1)

    # extract node, element, alpha
    pts = mesh_obj["node"]
    tri = mesh_obj["element"]

    """ 1. problem setup """
    anomaly = anomaly
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly, background=1.0)

    # draw
    fig, axes = plt.subplots(2, 1, constrained_layout=True)
    fig.set_size_inches(6, 4)

    ax = axes[0]
    ax.axis("equal")
    ax.set_title(r"Input $\Delta$ Conductivities")

    delta_perm = np.real(mesh_new["perm"] - mesh_obj["perm"])
    im = ax.tripcolor(
        pts[:, 0], pts[:, 1], tri, delta_perm, shading="flat", cmap=plt.cm.viridis
    )

    """ 2. FEM forward simulations """
    # setup EIT scan conditions
    # adjacent stimulation (el_dist=1), adjacent measures (step=1)
    el_dist, step = 1, 1
    ex_mat = eit_scan_lines(n_el, el_dist)

    # calculate simulated data
    fwd = Forward(mesh_obj, el_pos)
    f0 = fwd.solve_eit(ex_mat, step=step, perm=mesh_obj["perm"])  # default data without perturbation
    f1 = fwd.solve_eit(ex_mat, step=step, perm=mesh_new["perm"])  # data with perturbation

    """
    3. naive inverse solver using back-projection
    """
    eit = bp.BP(mesh_obj, el_pos, ex_mat=ex_mat, step=1, parser="std")
    eit.setup(weight="none")
    ds = eit.solve(f1.v, f0.v, normalize=False)

    # plot
    ax1 = axes[1]
    im = ax1.tripcolor(pts[:, 0], pts[:, 1], tri, ds, cmap=plt.cm.viridis)
    ax1.set_title(r"Reconstituted $\Delta$ Conductivities")
    ax1.axis("equal")
    fig.colorbar(im, ax=axes.ravel().tolist())


def gauss_newtonian(n_el, anomaly, regularization=False, plot=True, raw_measures=False, return_delta_perm=False,
                    verbose=True):
    """ 0. construct mesh """
    # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax , Default :fd=circle
    mesh_obj, el_pos = mesh.create(n_el, h0=0.1, fd=circle)

    # extract node, element, alpha
    pts = mesh_obj["node"]
    tri = mesh_obj["element"]
    x, y = pts[:, 0], pts[:, 1]

    print(len(x), len(y), len(tri))
    exit(0)

    """ 1. problem setup """
    anomaly = anomaly
    mesh_obj["alpha"] = np.random.rand(tri.shape[0]) * 200 + 100
    mesh_new = mesh.set_perm(mesh_obj, anomaly=anomaly)

    if return_delta_perm and not raw_measures:
        return np.array([]), mesh_new["perm"] - mesh_obj["perm"]

    """ 2. FEM simulation """
    el_dist, step = 1, 1
    ex_mat = eit_scan_lines(n_el, el_dist)

    # calculate simulated data
    fwd = Forward(mesh_obj, el_pos)
    f0 = fwd.solve_eit(ex_mat, step=step, perm=mesh_obj["perm"])
    f1 = fwd.solve_eit(ex_mat, step=step, perm=mesh_new["perm"])

    if raw_measures and not return_delta_perm:
        return f1.v
    elif raw_measures and return_delta_perm:
        return f1.v, mesh_new["perm"] - mesh_obj["perm"]

    start_time = datetime.now()
    if regularization:
        """ 3. solve_eit using gaussian-newton (with regularization) """
        # number of stimulation lines/patterns
        eit = jac.JAC(mesh_obj, el_pos, ex_mat, step, perm=1.0, parser="std")
        eit.setup(p=0.25, lamb=1.0, method="lm")
        # lamb = lamb * lamb_decay
        ds_n = eit.gn(f1.v, lamb_decay=0.1, lamb_min=1e-5, maxiter=20, verbose=verbose)
    else:
        """ 3. JAC solver """
        # Note: if the jac and the real-problem are generated using the same mesh,
        # then, data normalization in solve are not needed.
        # However, when you generate jac from a known mesh, but in real-problem
        # (mostly) the shape and the electrode positions are not exactly the same
        # as in mesh generating the jac, then data must be normalized.
        eit = jac.JAC(
            mesh_obj,
            el_pos,
            ex_mat=ex_mat,
            step=step,
            perm=1.0,
            parser="std",
        )
        eit.setup(p=0.5, lamb=0.01, method="kotre")
        ds = eit.solve(f1.v, f0.v, normalize=True)
        ds_n = sim2pts(pts, tri, np.real(ds))

    if not plot:
        return f1.v, ds_n
    else:
        fig, axes = plt.subplots(1, 2, constrained_layout=True)

        fig.set_size_inches(6, 4)

        # plot ground truth
        ax = axes[0]
        ax.set_title('Ground truth')
        delta_perm = mesh_new["perm"] - mesh_obj["perm"]
        im = ax.tripcolor(x, y, tri, np.real(delta_perm), shading="flat")
        ax.set_aspect("equal")

        # plot EIT reconstruction
        ax = axes[1]
        ax.set_title(f'Gauss-newtonian reconstruction ({regularization=})')
        im = ax.tripcolor(x, y, tri, ds_n, shading="flat")
        ax.set_aspect("equal")

        fig.colorbar(im, ax=axes.ravel().tolist())


def gauss_newtonian_extract_image(x, y, tri, ds_n, grayscale=False):
    fig, ax = plt.subplots(1, 1,)

    fig.set_size_inches(2.56, 2.56)
    im = ax.tripcolor(x, y, tri, ds_n, shading="flat")
    # Image from plot
    ax.axis('off')
    fig.tight_layout(pad=0)
    # To remove the huge white borders
    ax.margins(0)

    fig.canvas.draw()
    if grayscale:
        # Grayscale
        image_from_plot = np.array(fig.canvas.buffer_rgba())
        image_from_plot = np.rint(image_from_plot[..., :3] @ [0.2126, 0.7152, 0.0722]).astype(np.uint8)
    else:
        # Color
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image_from_plot


def main():
    db, collection = "EIT_datasets", "simulation_positive"
    mongoClient = mongodbAPI.MongodbAPI(db=db)
    mongoClient.setCollection(collection)

    # n_el = 64

    start_time = datetime.now()
    nb_sim = 1
    n_el_range = [8, 16, 32, 64]
    anomalies_list = []
    anomalies = [
        {"x": 0.5, "y": 0, "d": 0.2, "perm": 10},
        {"x": -0.5, "y": 0, "d": 0.2, "perm": 10}
    ]
    anomalies_list.append(anomalies)
    anomalies = [
        {"x": 0, "y": 0.5, "d": 0.2, "perm": 10},
        {"x": 0, "y": -0.5, "d": 0.2, "perm": 10},
    ]
    anomalies_list.append(anomalies)
    anomalies = [
        {"x": 0, "y": 0.5, "d": 0.2, "perm": 10},
        {"x": 0, "y": -0.5, "d": 0.2, "perm": 10},
        {"x": 0.5, "y": 0, "d": 0.2, "perm": 10},
        {"x": -0.5, "y": 0, "d": 0.2, "perm": 10}
    ]
    anomalies_list.append(anomalies)
    anomalies = [
        # left lung
        {"x": -0.6, "y": 0, "d": 0.2, "perm": 10},
        {"x": -0.5, "y": 0.2, "d": 0.2, "perm": 10},
        {"x": -0.5, "y": -0.2, "d": 0.2, "perm": 10},
        {"x": -0.4, "y": 0.4, "d": 0.2, "perm": 10},
        {"x": -0.4, "y": -0.4, "d": 0.2, "perm": 10},
        # right lung
        {"x": 0.6, "y": 0, "d": 0.2, "perm": 10},
        {"x": 0.5, "y": 0.2, "d": 0.2, "perm": 10},
        {"x": 0.5, "y": -0.2, "d": 0.2, "perm": 10},
        {"x": 0.4, "y": 0.4, "d": 0.2, "perm": 10},
        {"x": 0.4, "y": -0.4, "d": 0.2, "perm": 10},
    ]
    anomalies_list.append(anomalies)

    for _ in range(2100):
        lst = []
        for _ in range(randint(0, 5)):
            lst.append({"x": uniform(-1, 1), "y": uniform(-1, 1), "d": uniform(0.01, 0.75), "perm": uniform(1, 10)})
        anomalies_list.append(
            lst
        )

    data_to_append=[]
    for i, data in enumerate(anomalies_list):
        data_to_append.append({
            'anomalies': data,
            'image': i
        })

    print(mongoClient.insertData(data_to_append, collection))


def fill_in(chunk, input, n_el):
    chunk_calculated = []
    for p in chunk:
        v, d = gauss_newtonian(n_el, p["anomalies"],
                               regularization=True,
                               raw_measures=True,
                               plot=False,
                               return_delta_perm=True,
                               verbose=False)

        data = {"_id": p['_id'], f'v{n_el}': v.tolist(), f'd{n_el}': d.tolist()}
        chunk_calculated.append(data)
    return chunk_calculated


db, collection = "EIT_datasets", "simulation_positive"
mongoClient = mongodbAPI.MongodbAPI(db=db, collection=collection)


# takes a list and integer n as input and returns generator objects of n lengths from that list
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main1():
    n_el_range = [8, 16, 32, 64]

    for n in n_el_range:
        n_el = n
        print(n_el)
        query, option = {
                            f'delta_perm{n_el}': {'$exists': False},
                            f'v{n_el}': {'$exists': False},
                            # f'ds_n{n_el}': {'$exists': False},
                        }, {
                            f"anomalies": 1,
                            f"_id": 1
                        }

        cursor = mongoClient.collection.find(query, option).batch_size(5)

        cursor = list(cursor)
        chunksize = 100
        n = mongoClient.collection.count_documents(query)
        with Pool(processes=1) as pool:
            with tqdm(total=n) as pbar:
                calculate_partial = partial(fill_in,
                                            input=input,
                                            n_el=n_el)
                for u in pool.imap_unordered(calculate_partial, list(chunks(cursor, chunksize))):
                    pbar.update(len(u))
                    for p in u:
                        mongoClient.collection.update_one({
                            '_id': p['_id']
                        }, {
                            '$set': {
                                f'delta_perm{n_el}': p[f'd{n_el}'],
                                f'v{n_el}': p[f'v{n_el}'],
                                # f'ds_n{n_el}': p[f'd{n_el}'],
                            }
                        }, upsert=False)


def export_to_csv():
    db, collection = "EIT_datasets", "simulation_positive"
    mongoClient = mongodbAPI.MongodbAPI(db=db, collection=collection)
    mongoClient.export_collection_to_csv("dataset/eit_positive.csv")


def create_and_save_image(param, input, x, y, tri, path, delta_perm=True):
    for img in param:
        values = np.array(img[0])
        if delta_perm:
            values = np.real(values)
        image_name = img[1]
        im = gauss_newtonian_extract_image(x, y, tri, values, grayscale=False)
        imsave(join(path, f'{image_name}.png'), im)
    return [0] * len(param)


def export_images(path: str, csv_path: str, nb_elect: int):
    path_base = path

    """ 0. construct mesh """
    # Mesh shape is specified with fd parameter in the instantiation, e.g : fd=thorax , Default :fd=circle
    mesh_obj, el_pos = mesh.create(nb_elect, h0=0.1, fd=circle)
    # extract node, element, alpha
    pts = mesh_obj["node"]
    tri = mesh_obj["element"]
    x, y = pts[:, 0], pts[:, 1]

    df = pd.read_csv(csv_path, usecols=[f"delta_perm{nb_elect}", f"ds_n{nb_elect}", "image"],
                     converters={f"delta_perm{nb_elect}": literal_eval, f"ds_n{nb_elect}": literal_eval})
    print(df.info(memory_usage="deep"))
    img_list = [f"delta_perm{nb_elect}", f"ds_n{nb_elect}"]
    chunksize = 100
    n_proc = 4
    for img in img_list:
        n = len(df[img])
        path = join(path_base, img)
        if not exists(path):
            makedirs(path)

        with Pool(processes=n_proc) as pool:
            with tqdm(total=n) as pbar:
                calculate_partial = partial(create_and_save_image,
                                            input=input,
                                            x=x,
                                            y=y,
                                            tri=tri,
                                            path=path,
                                            delta_perm="delta_perm" in img)
                for u in pool.imap_unordered(calculate_partial, list(chunks(list(zip(df[img].tolist(), df[f"image"].tolist())), chunksize))):
                    pbar.update(len(u))


if __name__ == '__main__':
    main1()
    export_to_csv()
    for i in [8, 16, 32, 64]:
        export_images("../dataset/images", "dataset/eit_positive.csv", i)
