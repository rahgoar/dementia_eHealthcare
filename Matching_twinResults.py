# Best mapping based on the optimized reward. Optimization formulation.
from pulp import LpProblem,LpVariable,LpMaximize,lpSum,LpStatus,value

def mapping_labels(y_pred,y_label):
  M= construct_assignment_matrix(y_pred,y_label)

  prob = LpProblem("Max_Matching", LpMaximize)
  var_name_prefix = 'IsSelected'
  N = 5
  vars = LpVariable.dicts("Choice", (range(N), range(N)), cat="Binary")

  objective_func = lpSum([vars[i][j]*M[i][j] for i in range(N) for j in range(N)]), "Total_correct_classes"
  prob += objective_func
  for i in range(N):
    prob += lpSum([vars[i][j] for j in range(N)]) ==1 , f"row_{i}"

  for j in range(N):
    prob += lpSum([vars[i][j] for i in range(N)]) ==1, f"column_{j}"

  #prob.writeLP("MaximumMatching.lp")

  prob.solve()

  #print("Status:", LpStatus[prob.status])

  cluster_to_class = {}
  for v in prob.variables():
      if v.varValue>0.1: #working with float :)
        _,cluster,class_id = v.name.split('_')
        cluster_to_class[int(cluster)] = int(class_id)
  y_pred_class = np.zeros(y_pred.shape)
  for i,r in enumerate(y_pred):
    y_pred_class[i] = cluster_to_class[r]

  return y_pred_class, cluster_to_class
