# MaterialAgent

面向材料模拟任务的多 Agent 协作界面。

输入任务后，Planner 会制定计划，Worker 生成 LAMMPS（以及可选的 MATLAB）
脚本，Evaluator 负责审核与迭代。当工作流缺少关键参数或需要你确认时，
页面会弹出输入请求并暂停等待。
