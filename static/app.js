const form = document.getElementById('reservation');
const submit = document.getElementById('submit');
const prediction = document.getElementById('prediction');
const description = document.getElementById('description');
const area = document.getElementById('probability-area');
const error = document.getElementById('error');
const categorical = new Set(['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']);
function clearResult() {
  area.hidden = true;
  error.hidden = true;
  prediction.textContent = 'Pronto para testar';
  description.textContent = 'Clique em prever para consultar o modelo.';
}
document.getElementById('example').addEventListener('click', () => { form.reset(); clearResult(); });
form.addEventListener('input', clearResult);
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  clearResult();
  submit.disabled = true;
  prediction.textContent = 'Consultando o modelo…';
  const reservation = {};
  for (const [key, value] of new FormData(form)) {
    reservation[key] = categorical.has(key) ? value : Number(value);
  }
  try {
    const response = await fetch('/predict', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(reservation)
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.erro || 'Não foi possível realizar a previsão.');
    prediction.textContent = result.classe === 1 ? 'Cancelamento previsto' : 'Não cancelamento previsto';
    description.textContent = result.classe === 1 ? 'O modelo indicou que esta reserva pode ser cancelada.' : 'O modelo indicou que esta reserva pode ser mantida.';
    const percentage = result.probabilidade_cancelamento * 100;
    document.getElementById('probability').textContent = percentage.toLocaleString('pt-BR', {maximumFractionDigits: 2}) + '%';
    document.getElementById('bar').style.width = percentage + '%';
    area.hidden = false;
  } catch (failure) {
    prediction.textContent = 'Confira a reserva';
    description.textContent = 'A previsão não foi concluída.';
    error.textContent = failure.message;
    error.hidden = false;
  } finally { submit.disabled = false; }
});
