import os
import imageio
import random
import numpy as np
from tqdm import tqdm
from os.path import join
from operator import itemgetter
from itertools import permutations
import matplotlib.pyplot as plt


def get_distance(crd1, crd2):
    return np.sqrt((crd1[0] - crd2[0]) ** 2 + (crd1[1] - crd2[1]) ** 2)


def get_crds(env):
    pos_crds, imp_crds = [], []
    for i in range(len(env)):
        for j in range(len(env[0])):
            if env[i, j] == 0:
                pos_crds.append((i, j))
            else:
                imp_crds.append((i, j))
    return pos_crds, imp_crds


def get_candidates(env, crd, ways=None, died=None):
    if ways is None:
        ways = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    if died is None:
        died = []

    candidates = []
    for i, j in ways:
        x, y = crd[0]+i, crd[1]+j
        if (0 <= x < env.shape[0]) and (0 <= y < env.shape[1]) \
                and (env[x, y] == 0) and (x, y) not in died:
            candidates.append((x, y))
    return candidates


def random_cont(env, crd=None, lin=None, died=None):
    pos_crds, imp_crds = get_crds(env)
    if crd is None:
        crd = random.sample(pos_crds, 1)[0]
    elif crd in imp_crds:
        crd = random.sample(get_candidates(env, crd, died), 1)[0]

    if np.random.random() < .5:
        ways = [(0, 1), (0, -1)] if lin else None
    else:
        ways = [(1, 0), (-1, 0)] if lin else None

    if died is None:
        died_crds = [crd]
    else:
        died_crds = died + [crd]

    while len(died_crds) < len(pos_crds):
        candidates = get_candidates(env, crd, ways, died_crds)
        while len(candidates) > 0:
            crd = random.sample(candidates, 1)[0]
            died_crds.append(crd)
            candidates = get_candidates(env, crd, ways, died_crds)

        if len(died_crds) == len(pos_crds):
            break
        remains = list(set(pos_crds) - set(died_crds))
        remains_distance = [get_distance(crd, rem) for rem in remains]
        crd = remains[np.argmin(remains_distance)]
        died_crds.append(crd)
    return died_crds


def calc_distance_sum(route):
    distance = 0
    for idx in range(len(route) - 1):
        distance += get_distance(route[idx], route[idx+1])
    return distance


def rank_routes(routes):
    rank_info = {i: calc_distance_sum(routes[i]) for i in range(len(routes))}
    ranked = sorted(rank_info.items(), key=itemgetter(1))
    ranked = [i[0] for i in ranked]
    return ranked, rank_info


def get_cix(route):
    cix = []
    px, py = route[0]
    for i, (x, y) in enumerate(route[1:]):
        if (abs(px-x) > 1) or (abs(py-y) > 1):
            cix.append(i+1)
        px, py = x, y
    return cix


def sfl(route):
    n = len(route) // 10
    splited = []
    for i in range(0, len(route), n):
        splited.append(route[i:(i+n)])

    sfled = []
    indexes = np.random.choice(range(len(splited)), len(splited), replace=False)
    for idx in indexes:
        sfled.extend(splited[idx])
    return sfled


def breed(pra, prb, cix_ratio=.8):
    cra, crb = [], []
    if np.random.random() < cix_ratio:
        cix = get_cix(pra)
        route_a_idx = np.random.choice(range(len(cix)))
        route_b_idx = np.random.choice(range(len(cix[route_a_idx:]))) + route_a_idx
        starts, ends = cix[route_a_idx], cix[route_b_idx]
    else:
        route_a = int(random.random() * len(pra))
        route_b = int(random.random() * len(prb))
        starts, ends = min(route_a, route_b), max(route_a, route_b)

    for i in range(starts, ends):
        cra.append(pra[i])

    crb = [i for i in prb if i not in cra]
    return cra + crb


def regeneration(env, route):
    cix = get_cix(route)
    if len(cix) < 1:
        return route
    elif len(cix) == 1:
        regen_idx = random.sample(range(0, cix[0]-1), 1)[0]
        route = random_cont(env, route[regen_idx], True, died=route[:regen_idx])
    else:
        _begin = 0 if len(cix) == 2 else cix[-3]-1
        regen_idx = random.sample(range(_begin, cix[-2]-1), 1)[0]
        route = random_cont(env, route[regen_idx], True, died=route[:regen_idx])
    return route


def mutate(route):
    swap_a = int(random.random() * len(route))
    swap_b = int(random.random() * len(route))

    sa, sb = route[swap_a], route[swap_b]
    route[swap_a], route[swap_b] = sb, sa
    return route


def confirm_aircut(pc, coo):
    na = [
        (pc[0], pc[1]+1), (pc[0], pc[1]-1), (pc[0]+1, pc[1]), (pc[0]-1, pc[1]),
        (pc[0]+1, pc[1]+1), (pc[0]+1, pc[1]-1), (pc[0]-1, pc[1]+1), (pc[0]-1, pc[1]-1)
    ]
    if coo not in na:
        return 1
    return 0


def save_images_rgb(data, solution, save_path):
    data3 = np.concatenate([data[..., np.newaxis] for _ in range(3)], axis=-1)

    plt.figure()
    plt.imshow(data3, vmin=0, vmax=1)
    plt.axis("off")
    plt.title("Time:0s    AirCut:0", fontsize=15)
    plt.savefig(join(save_path, "./f0.png"), bbox_inches="tight", pad_inches=0)
    plt.clf()

    px, py = solution[0][0], solution[0][1]
    data3[px, py, :] = [.9, .1, .1]

    plt.imshow(data3, vmin=0, vmax=1)
    plt.axis("off")
    plt.title("Time:0s    AirCut:0", fontsize=15)
    plt.savefig(join(save_path, "./f1.png"), bbox_inches="tight", pad_inches=0)
    plt.clf()

    t, ac = 1, 0
    prog = tqdm(
        enumerate(solution[1:]),
        total=len(solution[1:]),
        desc="visualizing..."
    )
    for ind, (x, y) in prog:
        t += get_distance((px, py), (x, y))
        ac += confirm_aircut((px, py), (x, y))

        if sum(data3[x, y, :]) == 0:
            data3[x, y, :] = [.9, .1, .1]
        else:
            data3[x, y, :] = [.9, .1, .1]  # [.0, .0, .3]
        data3[px, py, :] = [.7, .7, .7]

        px, py = x, y
        plt.imshow(data3, vmin=0, vmax=1)
        plt.axis("off")
        plt.title(f"Time:{t:.2f}s    AirCut:{int(ac)}", fontsize=15)
        plt.savefig(join(save_path, f"./f{ind+2}.png"), bbox_inches="tight", pad_inches=0)
        plt.clf()


def save_gif(duration=.1, loop=0, png_path="./", save_path="./route.gif"):
    images = []
    files = sorted(os.listdir(png_path), key=lambda x: int(x.split(".")[0][1:]))
    for f in files:
        images.append(imageio.imread(join(png_path, f)))
        os.remove(join(png_path, f))

    imageio.mimsave(save_path, images, duration=duration, loop=loop)


class InitRoute:
    def __init__(self, env):
        self.env = env
        self.pos_crds, self.imp_crds = get_crds(env)

        cont_routes = []
        for crd in self.pos_crds:
            cont_routes.extend(self.random_cont_generation(5, crd, True))
        self.n_routes = len(cont_routes)
        self.set_routes()

        cont_routes = []
        for crd in self.pos_crds:
            route = self.random_cont_generation(5, crd, True)
            cont_routes.extend(route)
        self.routes = cont_routes

    def random_generation(self, n_routes=None):
        if n_routes is None:
            n_routes = self.n_routes

        routes = []
        for _ in range(n_routes):
            _route = random.sample(self.pos_crds, len(self.pos_crds))
            routes.append(_route)
        return routes

    def remove_crds(self, routes):
        return [crd for crd in routes if crd not in self.imp_crds]

    def random_cont_generation(self, n_routes, crd=None, lin=None):
        routes = []
        for _ in range(n_routes):
            if lin is None:
                lin = True if np.random.random() < .5 else None
            elif lin is False:
                lin = None
            routes.append(random_cont(self.env, crd, lin))
        return routes

    def set_routes(self):
        self.fg = self.random_generation(1)[0]
        self.tg = sfl(self.random_cont_generation(1, lin=False)[0])
        self.th = self.random_cont_generation(1, lin=False)[0]
        self.f = sfl(self.random_cont_generation(1, lin=True)[0])

    @staticmethod
    def breed(pg1, pg2):
        cg1, cg2 = [], []

        cg_part1 = int(random.random() * len(pg1))
        cg_part2 = int(random.random() * len(pg2))
        prg = range(min(cg_part1, cg_part2), max(cg_part1, cg_part2))
        for i in prg:
            cg1.append(pg1[i])

        cg2 = [i for i in pg2 if i not in cg1]
        return cg1 + cg2


class Solver:
    def __init__(self, env, n_routes, n_generations, elite_ratio=.5, mutate_ratio=.01, print_iter=1):
        self.env = env
        self.Initializer = InitRoute(env)
        self.n_routes = n_routes
        self.n_generations = n_generations
        self.elite_ratio = elite_ratio
        self.mutate_ratio = mutate_ratio
        self.print_iter = print_iter

        self.init_routes = self.Initializer.routes

    def train(self):
        routes, rank_info = self.build_next_generation(self.init_routes)
        rank_info = list(rank_info.values())
        # print(f"Initial - [Min:{min(rank_info):.3f}] [Mean:{np.mean(rank_info):.3f}]")

        prog = tqdm(
            range(1, self.n_generations),
            total=self.n_generations-1,
            desc="training..."
        )
        # for i in range(1, self.n_generations):
        for i in prog:
            routes, rank_info = self.build_next_generation(routes)
            rank_info = list(rank_info.values())
            # if i % self.print_iter == 0:
                # rank_info = list(rank_info.values())
                # print(f"{i} iterations - [Min:{min(rank_info):.3f}] [Mean:{np.mean(rank_info):.3f}]")
            prog.set_postfix(
                Min=min(rank_info),
                Mean=np.mean(rank_info)
            )

        self.routes = routes

    def build_next_generation(self, routes):
        ranked, rank_info = rank_routes(routes)
        # parent, child = self.breed_routes(routes, ranked, self.elite_ratio)
        parent, child = self.regen_routes(self.env, routes, ranked, self.elite_ratio)
        new_routes = self.mutate_routes(parent+child, self.mutate_ratio)
        return new_routes, rank_info

    def get_routes(self, idx, extend=None):
        ranked, rank_info = rank_routes(self.routes)
        solution = self.routes[ranked[idx]]
        score = rank_info[ranked[idx]]
        if extend:
            return (self.Initializer.fg,
                    self.Initializer.tg,
                    self.Initializer.th,
                    self.Initializer.f,
                    solution), score,
        else:
            return solution, score

    @staticmethod
    def breed_routes(routes, ranked, n_elite=None):
        if n_elite is None:
            n_elite = len(ranked) // 2
        if isinstance(n_elite, float):
            n_elite = int(len(ranked) * n_elite)

        pool_indexes = ranked[:n_elite]
        parents = [routes[i] for i in pool_indexes]
        parents_perm = list(permutations(pool_indexes, 2))
        parents_perm = random.sample(parents_perm, n_elite)

        child = []
        for prai, prbi in parents_perm:
            child.append(breed(routes[prai], routes[prbi]))
        return parents, child

    @staticmethod
    def regen_routes(env, routes, ranked, n_elite=None):
        if n_elite is None:
            n_elite = int(len(ranked) * 0.2)
        else:
            n_elite = int(len(ranked) * n_elite)
        n_regen = len(ranked) - n_elite

        pool_indexes = ranked[:n_elite]
        parents = [routes[i] for i in pool_indexes]
        child = []
        for _ in range(n_regen):
            sample = random.sample(parents, 1)[0]
            child.append(regeneration(env, sample))
        return parents, child

    @staticmethod
    def mutate_routes(routes, ratio=.01):
        mutated = [routes[0]]
        for route in routes[1:]:
            if np.random.random() < ratio:
                mutated.append(mutate(route))
            else:
                mutated.append(route)
        return mutated
