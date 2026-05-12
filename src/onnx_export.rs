// ONNX-Export für das flache Sequential-Modell (Embedding → mLSTMBlock → LinearNoBias).
//
// Exportiert einen einzelnen Zeitschritt mit explizitem State-I/O:
//
//   Inputs:  token_id []int64,  c_state [H·dhv·dqk]f32,
//            n_state [H·dqk]f32,  m_state [H]f32
//   Outputs: logits [vocab]f32,  c_new, n_new, m_new (gleiche Shapes wie Inputs)
//
// Das erzeugte .onnx-File kann mit OpenVINO direkt auf der NPU/iGPU ausgeführt werden.
// Keine externen Dependencies — der Protobuf-Encoder ist von Hand geschrieben.

use std::io;

use iron_oxide::collections::Matrix;

use crate::{
    nn::{embedding::EmbeddingLayer, linear_nb::LinearNBLayer, mlstm_block::MLSTMBlock},
    sequential::Sequential,
};

// ─── Minimaler Protobuf-Encoder ───────────────────────────────────────────────
// Feldnummern und Wire-Typen folgen dem ONNX-Proto-Schema (onnx-ml.proto).

struct Pb(Vec<u8>);

impl Pb {
    fn new() -> Self {
        Pb(Vec::new())
    }

    fn varint(&mut self, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                self.0.push(b);
                break;
            }
            self.0.push(b | 0x80);
        }
    }

    // Wire 0: varint (int32, int64, bool, enum)
    fn int64(&mut self, field: u32, v: i64) {
        self.varint(((field << 3) | 0) as u64);
        self.varint(v as u64);
    }

    // Wire 2: length-delimited (bytes, string, embedded message)
    fn bytes(&mut self, field: u32, data: &[u8]) {
        self.varint(((field << 3) | 2) as u64);
        self.varint(data.len() as u64);
        self.0.extend_from_slice(data);
    }

    fn string(&mut self, field: u32, s: &str) {
        self.bytes(field, s.as_bytes());
    }

    fn msg(&mut self, field: u32, m: &Pb) {
        self.bytes(field, &m.0);
    }

    // Repeated non-packed int64 (jeder Wert bekommt eigenen Tag — so schreibt ONNX dims)
    fn rep_i64(&mut self, field: u32, vals: &[i64]) {
        for &v in vals {
            self.int64(field, v);
        }
    }

    // Rohe f32-Bytes für raw_data-Felder im TensorProto
    fn raw_f32s(&mut self, field: u32, vals: &[f32]) {
        let mut bytes = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.bytes(field, &bytes);
    }

    // Rohe i64-Bytes für raw_data-Felder im TensorProto
    fn raw_i64s(&mut self, field: u32, vals: &[i64]) {
        let mut bytes = Vec::with_capacity(vals.len() * 8);
        for &v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.bytes(field, &bytes);
    }
}

// ─── ONNX-Proto-Bausteine ─────────────────────────────────────────────────────
//
// Feldnummern aus onnx-ml.proto (opset 13):
//   TensorProto: dims=1, data_type=2, name=8, raw_data=9
//   ValueInfoProto: name=1, type=2
//   TypeProto: tensor_type=1  |  TypeProto.Tensor: elem_type=1, shape=2
//   TensorShapeProto: dim=1  |  Dimension: dim_value=1
//   NodeProto: input=1, output=2, name=3, op_type=4, attribute=5
//   AttributeProto: name=1, i=3, f=4, t=5, floats=6, ints=7, type=20
//   AttributeType: FLOAT=1, INT=2, INTS=7
//   ModelProto: ir_version=1, opset_import=8, graph=7
//   GraphProto: node=1, name=2, initializer=4, input=11, output=12
//   OperatorSetIdProto: domain=1, version=2
//   DataType: FLOAT=1, INT64=7

const DTYPE_FLOAT: i64 = 1;
const DTYPE_INT64: i64 = 7;

fn tensor_f32(name: &str, dims: &[i64], data: &[f32]) -> Vec<u8> {
    let mut t = Pb::new();
    t.rep_i64(1, dims);
    t.int64(2, DTYPE_FLOAT);
    t.raw_f32s(9, data);
    t.string(8, name);
    t.0
}

fn tensor_i64(name: &str, dims: &[i64], data: &[i64]) -> Vec<u8> {
    let mut t = Pb::new();
    t.rep_i64(1, dims);
    t.int64(2, DTYPE_INT64);
    t.raw_i64s(9, data);
    t.string(8, name);
    t.0
}

fn value_info_f32(name: &str, dims: &[i64]) -> Vec<u8> {
    let mut shape = Pb::new();
    for &d in dims {
        let mut dim = Pb::new();
        dim.int64(1, d);
        shape.msg(1, &dim);
    }
    let mut tensor_type = Pb::new();
    tensor_type.int64(1, DTYPE_FLOAT);
    tensor_type.msg(2, &shape);
    let mut typ = Pb::new();
    typ.msg(1, &tensor_type);
    let mut vi = Pb::new();
    vi.string(1, name);
    vi.msg(2, &typ);
    vi.0
}

fn value_info_i64(name: &str, dims: &[i64]) -> Vec<u8> {
    let mut shape = Pb::new();
    for &d in dims {
        let mut dim = Pb::new();
        dim.int64(1, d);
        shape.msg(1, &dim);
    }
    let mut tensor_type = Pb::new();
    tensor_type.int64(1, DTYPE_INT64);
    tensor_type.msg(2, &shape);
    let mut typ = Pb::new();
    typ.msg(1, &tensor_type);
    let mut vi = Pb::new();
    vi.string(1, name);
    vi.msg(2, &typ);
    vi.0
}

fn attr_int(name: &str, v: i64) -> Vec<u8> {
    let mut a = Pb::new();
    a.string(1, name);
    a.int64(3, v);
    a.int64(20, 2); // type = INT
    a.0
}

fn attr_ints(name: &str, vals: &[i64]) -> Vec<u8> {
    let mut a = Pb::new();
    a.string(1, name);
    for &v in vals {
        a.int64(7, v);
    }
    a.int64(20, 7); // type = INTS
    a.0
}

// Wraps a serialised TensorProto as a TENSOR-type attribute named "value".
// Used to build inline Constant nodes.
fn attr_const_value(tensor_proto: &[u8]) -> Vec<u8> {
    let mut a = Pb::new();
    a.string(1, "value");
    a.bytes(5, tensor_proto); // field 5 = t (TensorProto)
    a.int64(20, 4); // type = TENSOR
    a.0
}

fn node(op: &str, inputs: &[&str], outputs: &[&str], name: &str, attrs: &[Vec<u8>]) -> Vec<u8> {
    let mut n = Pb::new();
    for i in inputs {
        n.string(1, i);
    }
    for o in outputs {
        n.string(2, o);
    }
    n.string(3, name);
    n.string(4, op);
    for a in attrs {
        n.bytes(5, a);
    }
    n.0
}

// ─── Graph-Builder ────────────────────────────────────────────────────────────

struct GraphBuilder {
    nodes: Vec<Vec<u8>>,
    inits: Vec<Vec<u8>>,
    // init_inputs: Initializer-Shapes auch als graph.input eintragen.
    // OpenVINO erwartet dass alle Tensoren in graph.input stehen — auch Gewichte.
    init_inputs: Vec<Vec<u8>>,
    inputs: Vec<Vec<u8>>,
    outputs: Vec<Vec<u8>>,
    counter: usize,
}

impl GraphBuilder {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            inits: Vec::new(),
            init_inputs: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            counter: 0,
        }
    }

    fn uid(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{}_{}", prefix, self.counter)
    }

    fn init_f32(&mut self, name: &str, dims: &[i64], data: &[f32]) {
        self.inits.push(tensor_f32(name, dims, data));
        self.init_inputs.push(value_info_f32(name, dims));
    }

    fn init_i64(&mut self, name: &str, dims: &[i64], data: &[i64]) {
        // Emit as a Constant node rather than a graph initializer.
        // Initializers in graph.input are treated as runtime inputs by OpenVINO
        // (causing garbage shape values); initializers NOT in graph.input are not
        // found in the graph cache.  Constant nodes bypass both problems: they are
        // true compile-time constants that OpenVINO folds without touching graph.input.
        let t = tensor_i64(name, dims, data);
        let attr = attr_const_value(&t);
        self.op("Constant", &[], &[name], &[attr]);
    }

    fn op(&mut self, op_type: &str, inputs: &[&str], outputs: &[&str], attrs: &[Vec<u8>]) {
        let name = self.uid(op_type);
        self.nodes
            .push(node(op_type, inputs, outputs, &name, attrs));
    }

    fn add_input_f32(&mut self, name: &str, dims: &[i64]) {
        self.inputs.push(value_info_f32(name, dims));
    }

    fn add_input_i64(&mut self, name: &str, dims: &[i64]) {
        self.inputs.push(value_info_i64(name, dims));
    }

    fn add_output_f32(&mut self, name: &str, dims: &[i64]) {
        self.outputs.push(value_info_f32(name, dims));
    }
}

// ─── Hilfsfunktion: Matrix → flacher Vec<f32> (row-major) ────────────────────

fn flat(m: &Matrix) -> Vec<f32> {
    let mut out = Vec::with_capacity(m.rows() * m.cols());
    for i in 0..m.rows() {
        out.extend_from_slice(&m[i]);
    }
    out
}

// ─── RMSNorm-Teilgraph ────────────────────────────────────────────────────────
//
// Formel: rms = sqrt(mean(x²) + ε)  |  out[i] = gamma[i] · x[i]/rms
//
// input:  Tensorname [n], gamma_init: Name des Gewichts-Initializers
// Rückgabe: Name des Output-Tensors

fn add_rms_norm(g: &mut GraphBuilder, input: &str, gamma_init: &str, p: &str) -> String {
    let sq = format!("{p}_sq");
    let msq = format!("{p}_msq");
    let eps_c = format!("{p}_eps");
    let msq_e = format!("{p}_msqe");
    let rms = format!("{p}_rms");
    let xhat = format!("{p}_xhat");
    let out = format!("{p}_out");

    g.init_f32(&eps_c, &[], &[1e-6]);

    g.op("Mul", &[input, input], &[&sq], &[]);
    g.op(
        "ReduceMean",
        &[&sq],
        &[&msq],
        &[attr_ints("axes", &[0]), attr_int("keepdims", 0)],
    );
    g.op("Add", &[&msq, &eps_c], &[&msq_e], &[]);
    g.op("Sqrt", &[&msq_e], &[&rms], &[]);
    g.op("Div", &[input, &rms], &[&xhat], &[]);
    g.op("Mul", &[gamma_init, &xhat], &[&out], &[]);

    out
}

// ─── mLSTM-Cell-Teilgraph ─────────────────────────────────────────────────────
//
// Gibt (cell_out, c_new, n_new, m_new) als Tensornamen zurück.

fn add_mlstm_cell(
    g: &mut GraphBuilder,
    input: &str,
    c_state: &str,
    n_state: &str,
    m_state: &str,
    cell: &crate::nn::mlstm::MLSTMLayer,
    p: &str,
) -> (String, String, String, String) {
    let hidden = cell.hidden_size;
    let h = cell.num_heads as i64;
    let dqk = cell.dqk as i64;
    let dhv = cell.dhv as i64;
    let d_qk = (cell.num_heads * cell.dqk) as i64;
    let c_flat = (cell.num_heads * cell.dhv * cell.dqk) as i64;
    let n_flat = d_qk;

    // Gewichte als Initializer
    macro_rules! wi {
        ($name:ident, $rows:expr, $cols:expr) => {
            let key = format!("{p}_{}", stringify!($name));
            g.init_f32(&key, &[$rows, $cols], &flat(&cell.$name));
        };
    }
    macro_rules! bi {
        ($name:ident, $len:expr) => {
            let key = format!("{p}_{}", stringify!($name));
            g.init_f32(&key, &[$len], &cell.$name);
        };
    }

    wi!(wq, hidden as i64, d_qk);
    wi!(wk, hidden as i64, d_qk);
    wi!(wv, hidden as i64, hidden as i64);
    wi!(wo, hidden as i64, hidden as i64);
    wi!(wi, hidden as i64, h);
    wi!(wf, hidden as i64, h);
    bi!(bq, d_qk);
    bi!(bk, d_qk);
    bi!(bv, hidden as i64);
    bi!(bo, hidden as i64);
    bi!(bi, h);
    bi!(bf, h);

    let wout_k = format!("{p}_wout");
    let bout_k = format!("{p}_bout");
    g.init_f32(
        &wout_k,
        &[hidden as i64, hidden as i64],
        &flat(&cell.w_out.weights),
    );
    g.init_f32(&bout_k, &[hidden as i64], &cell.w_out.biases);

    // Form-Konstanten für Reshape/Unsqueeze
    let sh_h_dqk = format!("{p}_sh_H_dqk");
    let sh_h_dhv = format!("{p}_sh_H_dhv");
    let sh_h_dhv_dqk = format!("{p}_sh_H_dhv_dqk");
    let sh_h_1_1 = format!("{p}_sh_H_1_1");
    let sh_h_1 = format!("{p}_sh_H_1");
    let sh_h_dhv_1 = format!("{p}_sh_H_dhv_1");
    let sh_h_1_dqk = format!("{p}_sh_H_1_dqk");
    let sh_h_dqk_1 = format!("{p}_sh_H_dqk_1");
    let sh_hidden = format!("{p}_sh_hidden");
    let sh_c_flat = format!("{p}_sh_c_flat");
    let sh_n_flat = format!("{p}_sh_n_flat");
    g.init_i64(&sh_h_dqk, &[2], &[h, dqk]);
    g.init_i64(&sh_h_dhv, &[2], &[h, dhv]);
    g.init_i64(&sh_h_dhv_dqk, &[3], &[h, dhv, dqk]);
    g.init_i64(&sh_h_1_1, &[3], &[h, 1, 1]);
    g.init_i64(&sh_h_1, &[2], &[h, 1]);
    g.init_i64(&sh_h_dhv_1, &[3], &[h, dhv, 1]);
    g.init_i64(&sh_h_1_dqk, &[3], &[h, 1, dqk]);
    g.init_i64(&sh_h_dqk_1, &[3], &[h, dqk, 1]);
    g.init_i64(&sh_hidden, &[1], &[hidden as i64]);
    g.init_i64(&sh_c_flat, &[1], &[c_flat]);
    g.init_i64(&sh_n_flat, &[1], &[n_flat]);

    let ones_h = format!("{p}_ones_h");
    g.init_f32(&ones_h, &[h], &vec![1.0f32; cell.num_heads]);

    let inv_sqrt = format!("{p}_inv_sqrt");
    g.init_f32(&inv_sqrt, &[], &[cell.inv_sqrt_dqk]);

    // Linearprojektionen: q, k, v, o, i_pre, f_pre
    macro_rules! proj {
        ($out:ident, $w:ident, $b:ident) => {
            let $out = format!("{p}_{}", stringify!($out));
            let raw = format!("{}_raw", $out);
            let wk = format!("{p}_{}", stringify!($w));
            let bk = format!("{p}_{}", stringify!($b));
            g.op("MatMul", &[input, &wk], &[&raw], &[]);
            g.op("Add", &[&raw, &bk], &[$out.as_str()], &[]);
        };
    }
    proj!(q, wq, bq);
    proj!(v, wv, bv);
    proj!(i_pre, wi, bi);
    proj!(f_pre, wf, bf);

    // k = (Wk·x + bk) · inv_sqrt_dqk
    let k_raw2 = format!("{p}_k_raw2");
    let k_raw3 = format!("{p}_k_raw3");
    let k = format!("{p}_k");
    let wk_k = format!("{p}_wk");
    let bk_k = format!("{p}_bk");
    g.op("MatMul", &[input, &wk_k], &[&k_raw2], &[]);
    g.op("Add", &[&k_raw2, &bk_k], &[&k_raw3], &[]);
    g.op("Mul", &[&k_raw3, &inv_sqrt], &[&k], &[]);

    // o = sigmoid(Wo·x + bo)
    let o_raw2 = format!("{p}_o_raw2");
    let o_raw3 = format!("{p}_o_raw3");
    let o = format!("{p}_o");
    let wo_k = format!("{p}_wo");
    let bo_k = format!("{p}_bo");
    g.op("MatMul", &[input, &wo_k], &[&o_raw2], &[]);
    g.op("Add", &[&o_raw2, &bo_k], &[&o_raw3], &[]);
    g.op("Sigmoid", &[&o_raw3], &[&o], &[]);

    // log_sigmoid(f_pre) = -Softplus(-f_pre)
    let f_neg = format!("{p}_f_neg");
    let sp = format!("{p}_sp");
    let log_f = format!("{p}_log_f");
    g.op("Neg", &[&f_pre], &[&f_neg], &[]);
    g.op("Softplus", &[&f_neg], &[&sp], &[]);
    g.op("Neg", &[&sp], &[&log_f], &[]);

    // Stabilizer: m_new = max(log_f + m_prev, i_pre)
    let lf_plus_m = format!("{p}_lf_plus_m");
    let m_new = format!("{p}_m_new");
    let i_minus_m = format!("{p}_i_minus_m");
    let lf_m_minus_m = format!("{p}_lf_m_minus_m");
    let i_prime = format!("{p}_i_prime");
    let f_prime = format!("{p}_f_prime");
    g.op("Add", &[&log_f, m_state], &[&lf_plus_m], &[]);
    g.op("Max", &[&lf_plus_m, &i_pre], &[&m_new], &[]);
    g.op("Sub", &[&i_pre, &m_new], &[&i_minus_m], &[]);
    g.op("Exp", &[&i_minus_m], &[&i_prime], &[]);
    g.op("Sub", &[&lf_plus_m, &m_new], &[&lf_m_minus_m], &[]);
    g.op("Exp", &[&lf_m_minus_m], &[&f_prime], &[]);

    // Per-Head-Reshape
    let q_h = format!("{p}_q_h");
    let k_h = format!("{p}_k_h");
    let v_h = format!("{p}_v_h");
    let o_h = format!("{p}_o_h");
    let c_h = format!("{p}_c_h");
    let n_h = format!("{p}_n_h");
    g.op("Reshape", &[&q, &sh_h_dqk], &[&q_h], &[]);
    g.op("Reshape", &[&k, &sh_h_dqk], &[&k_h], &[]);
    g.op("Reshape", &[&v, &sh_h_dhv], &[&v_h], &[]);
    g.op("Reshape", &[&o, &sh_h_dhv], &[&o_h], &[]);
    g.op("Reshape", &[c_state, &sh_h_dhv_dqk], &[&c_h], &[]);
    g.op("Reshape", &[n_state, &sh_h_dqk], &[&n_h], &[]);

    // Skalare für Broadcast: [H] → [H,1,1] und [H,1]
    let ip3 = format!("{p}_ip3");
    let fp3 = format!("{p}_fp3");
    let ip2 = format!("{p}_ip2");
    let fp2 = format!("{p}_fp2");
    g.op("Reshape", &[&i_prime, &sh_h_1_1], &[&ip3], &[]);
    g.op("Reshape", &[&f_prime, &sh_h_1_1], &[&fp3], &[]);
    g.op("Reshape", &[&i_prime, &sh_h_1], &[&ip2], &[]);
    g.op("Reshape", &[&f_prime, &sh_h_1], &[&fp2], &[]);

    // Äußeres Produkt v_h ⊗ k_h → [H, dhv, dqk]
    let v_uu = format!("{p}_v_uu");
    let k_uu = format!("{p}_k_uu");
    let outer = format!("{p}_outer");
    g.op("Reshape", &[&v_h, &sh_h_dhv_1], &[&v_uu], &[]); // [H,dhv,1]
    g.op("Reshape", &[&k_h, &sh_h_1_dqk], &[&k_uu], &[]); // [H,1,dqk]
    g.op("MatMul", &[&v_uu, &k_uu], &[&outer], &[]); // [H,dhv,dqk]

    // C-State-Update
    let fc = format!("{p}_fc");
    let io = format!("{p}_io");
    let c_new_h = format!("{p}_c_new_h");
    g.op("Mul", &[&fp3, &c_h], &[&fc], &[]);
    g.op("Mul", &[&ip3, &outer], &[&io], &[]);
    g.op("Add", &[&fc, &io], &[&c_new_h], &[]);

    // n-State-Update
    let fn_ = format!("{p}_fn");
    let ik = format!("{p}_ik");
    let n_new_h = format!("{p}_n_new_h");
    g.op("Mul", &[&fp2, &n_h], &[&fn_], &[]);
    g.op("Mul", &[&ip2, &k_h], &[&ik], &[]);
    g.op("Add", &[&fn_, &ik], &[&n_new_h], &[]);

    // cq = C_new @ q  (batched)
    let q_uu = format!("{p}_q_uu");
    let cq3 = format!("{p}_cq3");
    let cq = format!("{p}_cq");
    g.op("Reshape", &[&q_h, &sh_h_dqk_1], &[&q_uu], &[]); // [H,dqk,1]
    g.op("MatMul", &[&c_new_h, &q_uu], &[&cq3], &[]); // [H,dhv,1]
    g.op("Reshape", &[&cq3, &sh_h_dhv], &[&cq], &[]); // [H,dhv]

    // nq = n_new · q (batched dot)
    let nq_prod = format!("{p}_nq_prod");
    let nq = format!("{p}_nq");
    let red_ax = format!("{p}_red_ax");
    g.init_i64(&red_ax, &[1], &[1]);
    g.op("Mul", &[&n_new_h, &q_h], &[&nq_prod], &[]);
    g.op(
        "ReduceSum",
        &[&nq_prod, &red_ax],
        &[&nq],
        &[attr_int("keepdims", 0)],
    );

    // psi = max(|nq|, 1)
    let nq_abs = format!("{p}_nq_abs");
    let psi = format!("{p}_psi");
    let psi2 = format!("{p}_psi2");
    g.op("Abs", &[&nq], &[&nq_abs], &[]);
    g.op("Max", &[&nq_abs, &ones_h], &[&psi], &[]);

    // h_tilde = cq / psi, h_out = o * h_tilde
    let h_tilde = format!("{p}_h_tilde");
    let h_out = format!("{p}_h_out");
    let h_flat = format!("{p}_h_flat");
    g.op("Reshape", &[&psi, &sh_h_1], &[&psi2], &[]); // [H,1]
    g.op("Div", &[&cq, &psi2], &[&h_tilde], &[]); // [H,dhv]
    g.op("Mul", &[&o_h, &h_tilde], &[&h_out], &[]); // [H,dhv]
    g.op("Reshape", &[&h_out, &sh_hidden], &[&h_flat], &[]); // [hidden]

    // Ausgabeprojektion
    let cell_proj = format!("{p}_proj");
    let cell_out = format!("{p}_cell_out");
    g.op("MatMul", &[&h_flat, &wout_k], &[&cell_proj], &[]);
    g.op("Add", &[&cell_proj, &bout_k], &[&cell_out], &[]);

    // States zurück auf flache Form bringen
    let c_new = format!("{p}_c_new");
    let n_new = format!("{p}_n_new");
    g.op("Reshape", &[&c_new_h, &sh_c_flat], &[&c_new], &[]);
    g.op("Reshape", &[&n_new_h, &sh_n_flat], &[&n_new], &[]);

    (cell_out, c_new, n_new, m_new)
}

// ─── mLSTMBlock-Teilgraph ─────────────────────────────────────────────────────

fn add_mlstm_block(
    g: &mut GraphBuilder,
    input: &str,
    c_state: &str,
    n_state: &str,
    m_state: &str,
    block: &MLSTMBlock,
    p: &str,
) -> (String, String, String, String) {
    let hidden = block.hidden_size;
    let up = block.up_size;

    // Pre-RMSNorm
    let pre_gamma = format!("{p}_pre_gamma");
    g.init_f32(&pre_gamma, &[hidden as i64], &block.pre_norm1.gamma);
    let pre_normed = add_rms_norm(g, input, &pre_gamma, &format!("{p}_prn"));

    // mLSTM-Zelle
    let (cell_out, c_new, n_new, m_new) = add_mlstm_cell(
        g,
        &pre_normed,
        c_state,
        n_state,
        m_state,
        &block.cell,
        &format!("{p}_cell"),
    );

    // Residual 1: z = input + cell_out
    let z = format!("{p}_z");
    g.op("Add", &[input, &cell_out], &[&z], &[]);

    // Post-RMSNorm
    let post_gamma = format!("{p}_post_gamma");
    g.init_f32(&post_gamma, &[hidden as i64], &block.pre_norm2.gamma);
    let post_normed = add_rms_norm(g, &z, &post_gamma, &format!("{p}_pon"));

    // SwiGLU-MLP: gate = SiLU(Wg·z), val = Wv·z, mixed = gate⊙val, down = Wd·mixed
    macro_rules! mlp_layer {
        ($wname:expr, $bname:expr, $rows:expr, $cols:expr, $w:expr, $b:expr) => {
            g.init_f32(&$wname, &[$rows as i64, $cols as i64], &flat(&$w));
            g.init_f32(&$bname, &[$cols as i64], &$b[..]);
        };
    }

    let wg = format!("{p}_wgate");
    let bg = format!("{p}_bgate");
    let wv = format!("{p}_wval");
    let bv = format!("{p}_bval");
    let wd = format!("{p}_wdown");
    let bd = format!("{p}_bdown");
    mlp_layer!(
        wg,
        bg,
        hidden,
        up,
        block.lin_gate.weights,
        block.lin_gate.biases
    );
    mlp_layer!(
        wv,
        bv,
        hidden,
        up,
        block.lin_value.weights,
        block.lin_value.biases
    );
    mlp_layer!(
        wd,
        bd,
        up,
        hidden,
        block.lin_down.weights,
        block.lin_down.biases
    );

    let gp_r = format!("{p}_gp_r");
    let gp = format!("{p}_gp");
    let gsig = format!("{p}_gsig");
    let gact = format!("{p}_gact");
    let vr = format!("{p}_vr");
    let v = format!("{p}_v");
    let mix = format!("{p}_mix");
    let dr = format!("{p}_dr");
    let d = format!("{p}_d");
    let out = format!("{p}_out");

    g.op("MatMul", &[&post_normed, &wg], &[&gp_r], &[]);
    g.op("Add", &[&gp_r, &bg], &[&gp], &[]);
    g.op("Sigmoid", &[&gp], &[&gsig], &[]);
    g.op("Mul", &[&gp, &gsig], &[&gact], &[]); // SiLU = x·σ(x)
    g.op("MatMul", &[&post_normed, &wv], &[&vr], &[]);
    g.op("Add", &[&vr, &bv], &[&v], &[]);
    g.op("Mul", &[&gact, &v], &[&mix], &[]);
    g.op("MatMul", &[&mix, &wd], &[&dr], &[]);
    g.op("Add", &[&dr, &bd], &[&d], &[]);

    // Residual 2: output = z + down
    g.op("Add", &[&z, &d], &[&out], &[]);

    (out, c_new, n_new, m_new)
}

// ─── Vollständiger ONNX-Graph-Export ──────────────────────────────────────────

/// Exportiert das flache `Sequential`-Modell als ONNX-File.
///
/// Erwartet genau 4 Layer: `EmbeddingLayer → MLSTMBlock → LinearNBLayer → Softmax`.
/// Das erzeugte ONNX-Modell macht einen Schritt pro Aufruf und gibt die
/// neuen States zurück. Inference-Schleife:
///
/// ```rust
/// let mut c = vec![0.0f32; c_size];
/// let mut n = vec![0.0f32; n_size];
/// let mut m = vec![0.0f32; num_heads];
/// loop {
///     run_onnx(token, &c, &n, &m) → (logits, c_new, n_new, m_new)
///     (c, n, m) = (c_new, n_new, m_new);
/// }
/// ```
pub fn export_flat_model(model: &Sequential, path: &str) -> io::Result<()> {
    let emb = model.layers[0]
        .as_any()
        .downcast_ref::<EmbeddingLayer>()
        .expect("layer 0 must be EmbeddingLayer");
    let block = model.layers[1]
        .as_any()
        .downcast_ref::<MLSTMBlock>()
        .expect("layer 1 must be MLSTMBlock");
    let lm_head = model.layers[2]
        .as_any()
        .downcast_ref::<LinearNBLayer>()
        .expect("layer 2 must be LinearNBLayer");

    let vocab = emb.weights.rows() as i64;
    let hidden = emb.weights.cols() as i64;
    let h = block.cell.num_heads;
    let dhv = block.cell.dhv;
    let dqk = block.cell.dqk;
    let c_size = (h * dhv * dqk) as i64;
    let n_size = (h * dqk) as i64;
    let h_i = h as i64;

    let mut g = GraphBuilder::new();

    // Laufzeit-Inputs
    // token_id als [1] — OpenVINO ov_tensor_create unterstützt keine 0-D Tensoren
    g.add_input_i64("token_id", &[1]);
    g.add_input_f32("c_state", &[c_size]);
    g.add_input_f32("n_state", &[n_size]);
    g.add_input_f32("m_state", &[h_i]);

    // Embedding-Gewichte
    g.init_f32("emb_w", &[vocab, hidden], &flat(&emb.weights));

    // 1. Embedding-Lookup: emb_w[token_id] → [1, hidden], dann Reshape → [hidden]
    g.init_i64("tok_sh_hidden", &[1], &[hidden]);
    g.op(
        "Gather",
        &["emb_w", "token_id"],
        &["x_raw"],
        &[attr_int("axis", 0)],
    );
    g.op("Reshape", &["x_raw", "tok_sh_hidden"], &["x"], &[]);

    // 2. mLSTMBlock
    let (block_out, c_new, n_new, m_new) =
        add_mlstm_block(&mut g, "x", "c_state", "n_state", "m_state", block, "blk");

    // 3. LM-Head (LinearNoBias): [hidden] → [vocab]
    g.init_f32("lm_w", &[hidden, vocab], &flat(&lm_head.weights));
    g.op("MatMul", &[&block_out, "lm_w"], &["logits"], &[]);

    // Fixe Output-Namen für States (erleichtert Inferenz-Code)
    g.op("Identity", &[&c_new], &["c_new"], &[]);
    g.op("Identity", &[&n_new], &["n_new"], &[]);
    g.op("Identity", &[&m_new], &["m_new"], &[]);

    // Outputs deklarieren
    g.add_output_f32("logits", &[vocab]);
    g.add_output_f32("c_new", &[c_size]);
    g.add_output_f32("n_new", &[n_size]);
    g.add_output_f32("m_new", &[h_i]);

    // ONNX-Modell serialisieren und schreiben
    std::fs::write(path, build_model(g))
}

fn build_model(g: GraphBuilder) -> Vec<u8> {
    // GraphProto
    let mut graph = Pb::new();
    for n in &g.nodes {
        graph.bytes(1, n);
    }
    graph.string(2, "flat_mlstm");
    for i in &g.inits {
        graph.bytes(4, i);
    }
    for inp in &g.init_inputs {
        graph.bytes(11, inp);
    }
    for inp in &g.inputs {
        graph.bytes(11, inp);
    }
    for out in &g.outputs {
        graph.bytes(12, out);
    }

    // OperatorSetIdProto: standard opset 13
    let mut opset = Pb::new();
    opset.string(1, ""); // leere Domain = Standard-ONNX-Ops
    opset.int64(2, 13);

    // ModelProto
    let mut model = Pb::new();
    model.int64(1, 8); // ir_version = 8
    model.msg(8, &opset);
    model.msg(7, &graph);
    model.0
}
