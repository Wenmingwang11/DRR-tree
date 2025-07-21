import math
import pickle
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict
import multiprocessing
import sys
import cv2
from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Point:
    def __init__(self, class_name, track_id, frames, x, y, new_attr1, new_attr2, new_attr3):
        self.class_name = class_name
        self.track_id = track_id
        self.frames = frames
        self.x = round(x, 1)
        self.y = round(y, 1)
        self.new_attr1 = new_attr1
        self.new_attr2 = new_attr2
        self.new_attr3 = new_attr3

    def distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __repr__(self):
        return f"({self.class_name}, {self.track_id}, {self.frames}, {self.x:.1f}, {self.y:.1f})"

    def __hash__(self):
        return hash((self.class_name, self.track_id, tuple(self.frames), self.x, self.y))

    def __eq__(self, other):
        return (self.class_name, self.track_id, tuple(self.frames), self.x, self.y) == (
            other.class_name, other.track_id, tuple(other.frames), other.x, other.y)

    @staticmethod
    def calculate_centroid(cluster):
        """计算聚类的几何中心点"""
        if not cluster:
            return None
        center_x = sum(p.x for p in cluster) / len(cluster)
        center_y = sum(p.y for p in cluster) / len(cluster)
        return Point(None, None, [], center_x, center_y, None, None, None)


class Node:
    def __init__(self, points=None, distance_threshold=100):
        self.points = points if points else []
        self.children = []
        self.distance_threshold = distance_threshold
        self.threshold = 0
        self.flag = 0
        self.update_threshold()  # 初始化时自动计算阈值
        self.mbr_center = self.calculate_mbr_center()

    def merge_same_name_points(self, points):
        name_dict = defaultdict(lambda: Point(None, None, [], 0, 0, None, None, None))
        for point in points:
            key = (point.class_name, point.track_id)
            if key in name_dict:
                existing_point = name_dict[key]
                existing_point.frames = sorted(set(existing_point.frames + point.frames))
                total_frames = len(existing_point.frames) + len(point.frames)
                existing_point.x = (existing_point.x * len(existing_point.frames) + point.x * len(
                    point.frames)) / total_frames
                existing_point.y = (existing_point.y * len(existing_point.frames) + point.y * len(
                    point.frames)) / total_frames
                existing_point.x = round(existing_point.x, 1)
                existing_point.y = round(existing_point.y, 1)
            else:
                name_dict[key] = Point(point.class_name, point.track_id, point.frames.copy(), point.x, point.y,
                                       point.new_attr1, point.new_attr2, point.new_attr3)
        return list(name_dict.values())

    def calculate_threshold(self, points):
        if self.is_leaf() and len(points) == 1:
            return self.distance_threshold
        max_point_distance = 0
        if len(points) >= 2:
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    dist = points[i].distance(points[j])
                    if dist > max_point_distance:
                        max_point_distance = dist
        max_child_threshold = 0
        if self.children:
            max_child_threshold = max(child.threshold for child in self.children)
        return max(max_point_distance, max_child_threshold)

    def update_threshold(self):
        # 计算不合并同名点时的阈值
        non_merged_threshold = self.calculate_threshold(self.points)

        if self.flag == 1:
            # 如果 flag 为 1，阈值就是不合并的阈值
            self.threshold = non_merged_threshold
        else:
            # 合并同名点
            merged_points = self.merge_same_name_points(self.points)
            # 计算合并同名点后的阈值
            merged_threshold = self.calculate_threshold(merged_points)

            if merged_threshold == non_merged_threshold:
                # 若合并前后阈值相等，阈值为合并后的阈值
                self.threshold = merged_threshold
            else:
                # 若合并前后阈值不相等，阈值为不合并的阈值，且 flag 设为 1
                self.threshold = non_merged_threshold
                self.flag = 1

    def calculate_mbr_center(self):
        if not self.points:
            return Point(None, None, [], 0, 0, None, None, None)
        min_x = min(p.x for p in self.points)
        max_x = max(p.x for p in self.points)
        min_y = min(p.y for p in self.points)
        max_y = max(p.y for p in self.points)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        return Point(None, None, [], center_x, center_y, None, None, None)

    def add_points(self, new_points):
        self.points.extend(new_points)
        self.update_threshold()
        self.mbr_center = self.calculate_mbr_center()

    def is_leaf(self):
        return len(self.children) == 0


class DRRtree:
    def __init__(self, points, distance_threshold):
        self.root = self.build_tree(points, distance_threshold)

    def build_tree(self, points, distance_threshold):

        # Step 1: Preprocess points by merging those with the same track_id and close spatial proximity

        start_time = time.time()
        preprocess_threshold = 25
        track_groups = defaultdict(list)

        for point in points:
            track_groups[point.track_id].append(point)
        for track_id, track_points in track_groups.items():
            merged_points = []
            i = 0
            while i < len(track_points):
                current_point = track_points[i]
                merged_frames = current_point.frames.copy()
                j = i + 1
                while j < len(track_points):
                    next_point = track_points[j]
                    dist = current_point.distance(next_point)
                    if dist <= preprocess_threshold:
                        merged_frames.extend(next_point.frames)
                        merged_frames = sorted(set(merged_frames))
                        j += 1
                    else:
                        break
                merged_point = Point(current_point.class_name, current_point.track_id, merged_frames, current_point.x,
                                     current_point.y, current_point.new_attr1, current_point.new_attr2,
                                     current_point.new_attr3)
                merged_points.append(merged_point)
                i = j
            track_groups[track_id] = merged_points
        all_merged_points = []
        for sub_points in track_groups.values():
            all_merged_points.extend(sub_points)

        # print(f"预处理后点的数量: {len(all_merged_points)}")
        step_times = {}
        step_times['Step 1'] = time.time() - start_time


        clusters = []
        if all_merged_points:
            clusters.append([all_merged_points[0]])
            for point in all_merged_points[1:]:
                for cluster in clusters:
                    if all(point.distance(p) <= distance_threshold for p in cluster):
                        cluster.append(point)
                # 如果不满足现有任何组的条件，创建一个新组
                if not any(all(point.distance(p) <= distance_threshold for p in cluster) for cluster in clusters):
                    clusters.append([point])

        # 合并并行计算结果
        # 合并并行计算结果（这里由于新逻辑没有并行，直接使用聚类结果）

        start_time = time.time()
        leaf_nodes = []
        for cluster in clusters:
            node = Node(cluster, distance_threshold)
            leaf_nodes.append(node)
        self.merged_leaf_nodes = leaf_nodes
        for node in leaf_nodes:
            if node.flag == 0:
                node.points = node.merge_same_name_points(node.points)
                node.update_threshold()


        time1=time.time()
        # Step 7: Build the tree by merging nodes hierarchically
        # Step 7: 动态调整树高度
        def calculate_max_dist(points):

            max_d = 0
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    dist = points[i].distance(points[j])
                    if dist > max_d:
                        max_d = dist
            return max_d
        def merge_nodes(current_nodes, next_threshold):
            new_nodes = []
            used = set()
            merge_groups = []

            # 尝试合并节点并计算合并后的阈值
            for i in range(len(current_nodes)):
                if i in used:
                    continue
                current_group = {i}
                for j in range(i + 1, len(current_nodes)):
                    if j in used:
                        continue
                    # 创建临时父节点用于计算合并后的阈值
                    temp_parent = Node(distance_threshold=next_threshold)
                    temp_parent.children = [current_nodes[idx] for idx in current_group.union({j})]
                    all_points = []
                    for node in temp_parent.children:
                        all_points.extend(node.points)
                    temp_parent.add_points(all_points)
                    temp_parent.update_threshold()

                    # 判断合并后的阈值是否满足条件
                    if temp_parent.threshold <= next_threshold:
                        current_group.add(j)
                        used.add(j)

                merge_groups.append(current_group)
                used.add(i)

            # 处理合并组
            for group in merge_groups:
                group_nodes = [current_nodes[idx] for idx in group]
                parent = Node(distance_threshold=next_threshold)
                parent.children = group_nodes
                all_points = []
                for node in group_nodes:
                    all_points.extend(node.points)
                parent.add_points(all_points)
                parent.update_threshold()

                # 父节点继承子节点的 flag
                parent.flag = any(node.flag == 1 for node in group_nodes)

                new_nodes.append(parent)

            # 处理未分组的节点
            for i in range(len(current_nodes)):
                if i not in used:
                    new_nodes.append(current_nodes[i])

            return new_nodes

        def merge_nodes_optimized(current_nodes, next_threshold):
            # 复制一份节点列表，因为我们会修改它
            nodes = list(current_nodes)

            # 1. 预计算所有节点对的合并成本 (合并后的threshold)
            # 使用一个优先队列（min-heap）来存储，方便快速找到成本最低的合并
            import heapq
            possible_merges = []

            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    # 模拟合并 i 和 j
                    merged_points = nodes[i].points + nodes[j].points
                    # 计算合并后的新 threshold (这是昂贵操作，但只在预计算阶段做)
                    merged_threshold = calculate_max_dist(merged_points)

                    if merged_threshold <= next_threshold:
                        # 将 (成本, 节点1索引, 节点2索引) 放入优先队列
                        heapq.heappush(possible_merges, (merged_threshold, i, j))

            # 建立一个映射，追踪节点索引到实际Node对象（或其在列表中的位置）
            # 并查集（Disjoint Set Union, DSU）是实现这个的完美数据结构
            parent = list(range(len(nodes)))

            def find(i):
                if parent[i] == i:
                    return i
                parent[i] = find(parent[i])
                return parent[i]

            def union(i, j):
                root_i = find(i)
                root_j = find(j)
                if root_i != root_j:
                    parent[root_j] = root_i
                    return True
                return False

            # 2. 迭代合并
            while possible_merges:
                # 取出当前成本最低的合并选项
                cost, i, j = heapq.heappop(possible_merges)

                # 使用并查集检查 i 和 j 是否已经在同一个组里了
                # 如果是，则跳过
                if find(i) == find(j):
                    continue

                # 如果成本依然满足阈值（理论上一定满足，因为我们预筛选过）
                # 并且 i 和 j 不在同一组
                if cost <= next_threshold:
                    # 合并 i 和 j
                    union(i, j)

            # 3. 根据并查集的结果构建最终的合并组
            merge_groups_map = defaultdict(list)
            for i in range(len(nodes)):
                root = find(i)
                merge_groups_map[root].append(i)  # 将原始索引加入其根节点的组

            final_groups = list(merge_groups_map.values())

            # 后续根据 final_groups 创建父节点...
            # (这部分代码与原版类似，只是输入从 merge_groups 变成了 final_groups)
            new_nodes = []
            for group_indices in final_groups:
                if not group_indices:
                    continue

                # 如果组内只有一个节点，直接保留原节点
                if len(group_indices) == 1:
                    new_nodes.append(current_nodes[group_indices[0]])
                    continue

                # 创建新的父节点
                parent = Node(distance_threshold=next_threshold)
                group_nodes = [current_nodes[idx] for idx in group_indices]
                parent.children = group_nodes

                all_points = []
                for node in group_nodes:
                    all_points.extend(node.points)
                parent.add_points(all_points)
                parent.update_threshold()  # 这一步是必要的，给父节点设置正确的属性

                parent.flag = any(node.flag == 1 for node in group_nodes)

                new_nodes.append(parent)

            return new_nodes

        root=Node()

        # 第一层合并，阈值为 2 倍的 distance_threshold
        first_level_nodes = merge_nodes_optimized(leaf_nodes, 3 * distance_threshold)
        if len(first_level_nodes) == 1:
            root=first_level_nodes[0]
        else:
            second_level_nodes = merge_nodes_optimized(first_level_nodes, 9 * distance_threshold)
            if len(second_level_nodes) == 1:
                root = second_level_nodes[0]
            else:
                # 创建根节点时不指定阈值倍数
                root = Node()
                root.children = second_level_nodes
                root.add_points([p for node in second_level_nodes for p in node.points])
                # 让根节点根据自身点和子节点情况计算阈值
                root.update_threshold()




        return root

    def find_object_info(self, target_track_id):
        # 存储目标物体出现的帧
        target_frames = set()
        # 存储在同一帧中与目标物体相距小于节点阈值的其他点信息及所在节点阈值
        close_points_info = defaultdict(list)

        def traverse(node):
            for point in node.points:
                if point.track_id == target_track_id:
                    # 记录目标物体出现的帧
                    target_frames.update(point.frames)
                    # 查找同一帧中与目标物体相距小于节点阈值的其他点
                    for other_point in node.points:
                        if other_point.track_id != target_track_id:
                            common_frames = set(point.frames).intersection(set(other_point.frames))
                            for frame in common_frames:
                                if point.distance(other_point) < node.threshold:
                                    close_points_info[frame].append((other_point, node.threshold))
            # 递归遍历子节点
            for child in node.children:
                traverse(child)

        # 跳过根节点，从根节点的子节点开始遍历
        for child in self.root.children:
            traverse(child)

        print(f"物体 {target_track_id} 出现的帧: {sorted(target_frames)}")
        if close_points_info:
            print("在以下帧中与该物体相距小于节点阈值的其他点信息及所在节点阈值:")
            for frame, point_threshold_pairs in sorted(close_points_info.items()):
                for point, threshold in point_threshold_pairs:
                    print(f"帧 {frame}: 点 {point}，所在节点阈值: {threshold:.1f}")
        else:
            print("未找到在同一帧中与该物体相距小于节点阈值的其他点。")

    def find_close_points_in_frames(self, points, distance_threshold=100):
        # 用于存储最终结果，键为 (点1编号, 点2编号)，值为它们相近的帧列表
        close_points_result = defaultdict(list)

        def traverse(node):
            # 如果当前节点阈值大于指定阈值，继续遍历子节点
            if node.threshold > distance_threshold:
                for child in node.children:
                    traverse(child)
            else:
                # 处理阈值小于等于指定阈值的节点
                node_points = node.points
                num_points = len(node_points)
                for i in range(num_points):
                    for j in range(i + 1, num_points):
                        point1 = node_points[i]
                        point2 = node_points[j]
                        common_frames = set(point1.frames).intersection(set(point2.frames))
                        for frame in common_frames:
                            if point1.distance(point2) < distance_threshold:
                                key = tuple(sorted([point1.track_id, point2.track_id]))
                                if frame not in close_points_result[key]:
                                    close_points_result[key].append(frame)

                # 如果是叶子节点，节点数大于1且阈值大于指定阈值，进行逐个排查
                if node.is_leaf() and num_points > 1 and node.threshold > distance_threshold:
                    for i in range(num_points):
                        for j in range(i + 1, num_points):
                            point1 = node_points[i]
                            point2 = node_points[j]
                            common_frames = set(point1.frames).intersection(set(point2.frames))
                            for frame in common_frames:
                                if point1.distance(point2) < distance_threshold:
                                    key = tuple(sorted([point1.track_id, point2.track_id]))
                                    if frame not in close_points_result[key]:
                                        close_points_result[key].append(frame)

        # 从根节点开始遍历
        traverse(self.root)

        # 从原始数据计算真实相近点对
        true_close_points = defaultdict(list)
        num_all_points = len(points)
        for i in range(num_all_points):
            for j in range(i + 1, num_all_points):
                point1 = points[i]
                point2 = points[j]
                common_frames = set(point1.frames).intersection(set(point2.frames))
                for frame in common_frames:
                    if point1.distance(point2) < distance_threshold:
                        key = tuple(sorted([point1.track_id, point2.track_id]))
                        if frame not in true_close_points[key]:
                            true_close_points[key].append(frame)

        # 计算准确率
        correct_pairs = 0
        total_pairs = 0
        for key in set(true_close_points.keys()).union(set(close_points_result.keys())):
            true_frames = set(true_close_points.get(key, []))
            predicted_frames = set(close_points_result.get(key, []))
            correct_pairs += len(true_frames.intersection(predicted_frames))
            total_pairs += len(true_frames.union(predicted_frames))

        accuracy = correct_pairs / total_pairs if total_pairs > 0 else 0

        # 输出结果
        if close_points_result:
            print("找到满足条件的物体对：")
            for (track_id1, track_id2), frames in close_points_result.items():
                frame_intervals = ', '.join(map(str, sorted(frames)))
                print(f"物体 {track_id1} 和物体 {track_id2} 在帧 {frame_intervals} 中相近。")
        else:
            print("未找到满足条件的物体对。")

        print(f"准确率: {accuracy * 100:.2f}%")

        return close_points_result

    def calculate_mbr_distance(self, mbr1, mbr2):
        """计算两个MBR之间的最小距离"""
        dx = max(mbr1['min_x'] - mbr2['max_x'], mbr2['min_x'] - mbr1['max_x'], 0)
        dy = max(mbr1['min_y'] - mbr2['max_y'], mbr2['min_y'] - mbr1['max_y'], 0)
        return math.sqrt(dx ** 2 + dy ** 2)

    def group_nodes(self, nodes, distance_threshold):
        groups = []
        used = set()
        sorted_nodes = sorted(nodes, key=lambda x: x.threshold)  # 按阈值排序

        # 计算所有节点对之间的距离，并存储为优先队列
        distances = []
        for i in range(len(sorted_nodes)):
            for j in range(i + 1, len(sorted_nodes)):
                center1 = sorted_nodes[i].mbr_center
                center2 = sorted_nodes[j].mbr_center
                distance = center1.distance(center2)
                distances.append((distance, i, j))

        # 按距离从小到大排序
        distances.sort(key=lambda x: x[0])

        for distance, i, j in distances:
            if i in used or j in used:
                continue  # 如果其中一个节点已经被分组，则跳过

            # 创建一个新组
            group = [sorted_nodes[i], sorted_nodes[j]]
            groups.append(group)
            used.add(i)
            used.add(j)

        # 如果还有未分组的节点，单独作为一组
        for i in range(len(sorted_nodes)):
            if i not in used:
                groups.append([sorted_nodes[i]])

        return groups

    def calculate_group_center(self, nodes):
        all_points = [p for node in nodes for p in node.points]
        if not all_points:
            return Point(None, None, [], 0, 0, None, None, None)
        min_x = min(p.x for p in all_points)
        max_x = max(p.x for p in all_points)
        min_y = min(p.y for p in all_points)
        max_y = max(p.y for p in all_points)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        return Point(None, None, [], center_x, center_y, None, None, None)

    def get_all_leaf_children(self, node, all_leaf_nodes):
        if node in all_leaf_nodes:
            return [node]
        leaf_children = []
        for child in node.children:
            leaf_children.extend(self.get_all_leaf_children(child, all_leaf_nodes))
        return leaf_children

    def print_tree(self, node=None, level=0):
        if node is None:
            node = self.root
        indent = "  " * level
        leaf_tag = " [Leaf]" if node.is_leaf() else ""
        print(f"{indent}Level {level}{leaf_tag}, Threshold: {node.threshold:.1f}, Points: {node.points}")
        for child in node.children:
            self.print_tree(child, level + 1)

    def count_leaf_nodes(self, node=None):
        if node is None:
            node = self.root
        if node.is_leaf():
            return 1
        return sum(self.count_leaf_nodes(child) for child in node.children)

    def count_all_nodes(self, node=None):
        """
        统计树中所有节点的数量
        :param node: 当前要统计的节点，默认为根节点
        :return: 节点数量
        """
        if node is None:
            node = self.root
        # 如果当前节点为空，返回 0
        if node is None:
            return 0
        # 统计当前节点及其所有子节点的数量
        return 1 + sum(self.count_all_nodes(child) for child in node.children)

    def get_leaf_nodes(self, node=None):
        if node is None:
            node = self.root
        if node.is_leaf():
            return [node]
        leaf_nodes = []
        for child in node.children:
            leaf_nodes.extend(self.get_leaf_nodes(child))
        return leaf_nodes

    def print_leaf_nodes(self):
        leaf_nodes = self.get_leaf_nodes()
        print("叶子节点信息：")
        for i, node in enumerate(leaf_nodes):
            # 计算当前叶子节点的 MBR
            mbr = self.calculate_cluster_mbr(node.points)
            print(f"Leaf Node {i + 1}: Threshold = {node.threshold:.1f}, Points = {node.points}")
            print(
                f"MBR: min_x = {mbr['min_x']:.1f}, max_x = {mbr['max_x']:.1f}, min_y = {mbr['min_y']:.1f}, max_y = {mbr['max_y']:.1f}")

    def draw_leaf_nodes(self):
        leaf_nodes = self.get_leaf_nodes()
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, node in enumerate(leaf_nodes):
            if not node.points:
                continue
            # 计算当前叶子节点的 MBR
            mbr = self.calculate_cluster_mbr(node.points)
            min_x = mbr['min_x']
            max_x = mbr['max_x']
            min_y = mbr['min_y']
            max_y = mbr['max_y']
            width = max_x - min_x
            height = max_y - min_y
            rect = Rectangle((min_x, min_y), width, height, edgecolor=colors[i % len(colors)],
                             facecolor='none')
            ax.add_patch(rect)
            # 绘制叶子节点中的 Point 点
            x_coords = [point.x for point in node.points]
            y_coords = [point.y for point in node.points]
            ax.scatter(x_coords, y_coords, c=colors[i % len(colors)], s=10)  # s 为点的大小

            # 添加叶子节点编号标注
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            ax.text(center_x, center_y, str(i + 1), ha='center', va='center', fontsize=12,
                    color=colors[i % len(colors)])

        ax.set_title('Leaf Nodes (MBR and Points) Visualization')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        ax.autoscale_view()
        plt.grid(True)
        plt.show()

    def calculate_cluster_mbr(self, cluster):
        """计算点集的最小外接矩形（MBR）"""
        if not cluster:
            return {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0}
        min_x = min(p.x for p in cluster)
        max_x = max(p.x for p in cluster)
        min_y = min(p.y for p in cluster)
        max_y = max(p.y for p in cluster)
        return {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y}


def read_pkl_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def parse_dataframe_to_points(df):
    points = []
    for index, row in df.iterrows():
        class_name = str(row['class'])
        track_id = int(row['id'])
        frame = int(row['frame'])
        x = row['center_x']
        y = row['center_y']
        # 检查列是否存在，修改获取属性的列名
        new_attr1 = row.get('new_attr1')
        new_attr2 = row.get('new_attr2')
        new_attr3 = row.get('new_attr3')
        point = Point(class_name, track_id, [frame], x, y, new_attr1, new_attr2, new_attr3)
        points.append(point)
    return points


def build_tree_for_window(window_points, distance_threshold):
    """
    为单个窗口的数据构建 DRR-tree 的辅助函数，并记录构建时间
    :param window_points: 单个窗口的点数据
    :param distance_threshold: 距离阈值
    :return: (构建好的DRRtree对象, 构建时间)
    """
    start_time = time.time()  # 开始计时
    tree = DRRtree(window_points, distance_threshold)
    build_time = time.time() - start_time  # 结束计时

    def get_deep_size(obj, seen=None):
        """递归计算对象及其所有引用的内存大小"""
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        if hasattr(obj, '__dict__'):
            size += get_deep_size(obj.__dict__, seen)

        if isinstance(obj, (list, tuple, set)):
            size += sum(get_deep_size(item, seen) for item in obj)

        if isinstance(obj, dict):
            size += sum(get_deep_size(k, seen) + get_deep_size(v, seen) for k, v in obj.items())

        return size

    # 计算树的内存占用
    memory_usage = get_deep_size(tree)

    return (tree, build_time, memory_usage)



def compress_frames(frames):
    if not frames:
        return []

    # 先去除重复并排序
    unique_frames = sorted(set(frames))
    compressed = []
    start = unique_frames[0]
    end = unique_frames[0]

    for i in range(1, len(unique_frames)):
        if unique_frames[i] == end + 1:
            end = unique_frames[i]
        else:
            if start == end:
                compressed.append(start)
            else:
                compressed.append((start, end))
            start = unique_frames[i]
            end = unique_frames[i]

    # 处理最后的帧或区间
    if start == end:
        compressed.append(start)
    else:
        compressed.append((start, end))

    return compressed


def build_inverted_index(points, window_frames):
    inverted_index = {}
    temp_index = {}

    # 窗口映射表 {窗口索引: (起始帧, 结束帧)}
    window_map = {i: (start, end) for i, (start, end) in enumerate(window_frames)}

    # 第一遍：按物体收集数据
    for i, point in enumerate(points):
        # 确定当前点所属窗口（基于首帧范围）
        window_num = None
        for win_idx, (start, end) in window_map.items():
            if point.frames and start <= point.frames[0] <= end:
                window_num = win_idx
                break

        # 跳过不属于任何窗口的点
        if window_num is None:
            continue

        # 构建物体的唯一标识键
        key = (point.new_attr1, point.new_attr2, point.new_attr3, point.track_id)

        # 初始化或更新临时索引
        if key not in temp_index:
            temp_index[key] = {
                "frames": set(point.frames),  # 自动去重
                "windows": {window_num}  # 窗口集合
            }
        else:
            temp_index[key]["frames"].update(point.frames)
            temp_index[key]["windows"].add(window_num)

    # 第二遍：构建嵌套索引
    for (attr1, attr2, attr3, track_id), data in temp_index.items():
        # 压缩帧区间（保持原版逻辑）
        frames = sorted(data["frames"])
        compressed = []
        if frames:
            start = frames[0]
            for i in range(1, len(frames)):
                if frames[i] != frames[i - 1] + 1:
                    # 单帧或连续帧处理
                    if start == frames[i - 1]:
                        compressed.append(start)
                    else:
                        compressed.append((start, frames[i - 1]))
                    start = frames[i]
            # 处理最后一段
            if start == frames[-1]:
                compressed.append(start)
            else:
                compressed.append((start, frames[-1]))

        # 构建与原版完全一致的四级嵌套结构
        if attr1 not in inverted_index:
            inverted_index[attr1] = {}
        if attr2 not in inverted_index[attr1]:
            inverted_index[attr1][attr2] = {}
        if attr3 not in inverted_index[attr1][attr2]:
            inverted_index[attr1][attr2][attr3] = {}

        inverted_index[attr1][attr2][attr3][track_id] = {
            "frames": compressed,
            "windows": sorted(data["windows"])  # 原版要求排序
        }

    return inverted_index

def tree_built_callback(result, index, lock, time_list, memory_list):
    """
    树构建完成的回调函数，用于收集时间和内存数据
    """
    tree, build_time, memory_usage = result  # 解包结果
    with lock:
        time_list.append(build_time)  # 记录构建时间
        memory_list.append(memory_usage)  # 记录内存占用


def segment_video_by_oc_from_df(df, theta=0.6):
    """
    直接从DataFrame分割窗口（避免转换为Point对象）
    返回: [(start_frame, end_frame), ...]
    """
    # 按帧分组获取每帧的物体ID集合
    frame_groups = df.groupby('frame')['id'].apply(set)
    frames = sorted(frame_groups.index)

    if len(frames) == 0:
        return []

    windows = []
    current_start = frames[0]
    current_ids = frame_groups[current_start]

    for frame in frames[1:]:
        new_ids = frame_groups[frame]

        # 计算重叠系数
        common_ids = current_ids & new_ids
        min_len = min(len(current_ids), len(new_ids))
        oc = len(common_ids) / min_len if min_len > 0 else 0

        if oc < theta:
            windows.append((current_start, frame - 1))
            current_start = frame
            current_ids = new_ids

    # 添加最后一个窗口
    windows.append((current_start, frames[-1]))
    return windows

if __name__ == "__main__":
    pkl_file_path = 'modified_file3.pkl'  # 替换为实际的PKL文件路径
    df = read_pkl_file(pkl_file_path)

    window_frames = segment_video_by_oc_from_df(df, theta=0.5)
    print(f"分割结果（共{len(window_frames)}个窗口）:")
    for i, (start, end) in enumerate(window_frames):
        print(f"窗口{i + 1}: 帧{start}-{end} (共{end - start + 1}帧)")

    # points = parse_dataframe_to_points(data)
    distance_threshold = 100 # 根据实际需求设置
    # 记录开始时间
    start_time = time.time()
    # 准备每个窗口的数据
    window_data = []
    for start, end in window_frames:
        # 提取当前窗口的DataFrame数据
        window_df = df[(df['frame'] >= start) & (df['frame'] <= end)]
        # 转换为Point对象列表（保持与原有代码兼容）
        window_points = [
            Point(
                str(row['class']),
                int(row['id']),
                [int(row['frame'])],
                row['center_x'],
                row['center_y'],
                row.get('new_attr1'),
                row.get('new_attr2'),
                row.get('new_attr3')
            ) for _, row in window_df.iterrows()
        ]
        window_data.append((window_points, distance_threshold))

    # 用于存储所有窗口的树信息和统计数据
    all_trees = []
    build_times = []  # 存储所有树的构建时间
    tree_memory_usage_bytes = []  # 存储所有树的内存占用(字节)
    # 创建一个锁对象
    lock = multiprocessing.Lock()

    with multiprocessing.Pool() as pool:
        results = []
        for i, (window_points, dist_threshold) in enumerate(window_data):
            # 传递 build_times 和 tree_memory_usages 到回调函数
            result = pool.apply_async(
                build_tree_for_window,
                args=(window_points, dist_threshold),
                callback=lambda res, idx=i, l=lock, t=build_times, m=tree_memory_usage_bytes:
                tree_built_callback(res, idx, l, t, m)
            )
            results.append(result)

        # 等待所有任务完成
        for result in results:
            tree, _, _ = result.get()
            all_trees.append(tree)

    memory_usage_kb = [usage / 1024 for usage in tree_memory_usage_bytes]

    # 计算总内存占用(MB)
    total_memory_bytes = sum(tree_memory_usage_bytes)
    total_memory_mb = total_memory_bytes / (1024 * 1024)

    # 输出结果
    print("\n==== 每棵树内存占用统计 (KB) ====")

    # 额外统计：最大值、最小值和平均值
    if memory_usage_kb:
        max_kb = max(memory_usage_kb)
        min_kb = min(memory_usage_kb)
        avg_kb = sum(memory_usage_kb) / len(memory_usage_kb)

        print(f"\n统计信息:")
        print(f"最大树内存占用: {max_kb:.2f} KB")
        print(f"最小树内存占用: {min_kb:.2f} KB")
        print(f"平均树内存占用: {avg_kb:.2f} KB")
        print(f"总内存占用: {total_memory_mb:.2f} MB")

    # 基于整个数据集构建倒排索引
    all_points = [point for win_points, _ in window_data for point in win_points]
    start1_time = time.time()
    inverted_index = build_inverted_index(all_points,window_frames)
    total1_time = time.time() - start1_time
    print(f"倒排索引时间: {total1_time:.4f} 秒")


    def get_deep_size(obj, seen=None):
        """递归计算对象及其所有引用的内存大小"""
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        if hasattr(obj, '__dict__'):
            size += get_deep_size(obj.__dict__, seen)

        if isinstance(obj, (list, tuple, set)):
            size += sum(get_deep_size(item, seen) for item in obj)

        if isinstance(obj, dict):
            size += sum(get_deep_size(k, seen) + get_deep_size(v, seen) for k, v in obj.items())

        return size


    # 计算倒排索引的内存占用
    inverted_index_bytes = get_deep_size(inverted_index)
    inverted_index_kb = inverted_index_bytes / 1024
    inverted_index_mb = inverted_index_bytes / (1024 * 1024)
    print(f"倒排索引内存占用: {inverted_index_kb:.2f} KB ({inverted_index_mb:.4f} MB)")
    # 记录结束时间
    end_time = time.time()
    total_time = end_time - start_time

    # 计算倒排索引的内存大小
    index_memory_size = sys.getsizeof(inverted_index)

    # 统计构建时间指标
    if build_times:
        max_time = max(build_times)
        min_time = min(build_times)

        # 统计内存指标        print("\n==== DRR-tree 构建统计 ====")
        print(f"最大构建时间: {max_time:.4f} 秒")
        print(f"最小构建时间: {min_time:.4f} 秒")
        print(sum(build_times)/len(build_times))



    save_path = 'data/每层比值/A.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump({
            "trees": all_trees,
            "inverted_index": inverted_index
        }, f)


    print(f"建立完整个索引的总时间: {total_time} 秒")
