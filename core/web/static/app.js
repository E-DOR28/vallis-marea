/* Vallis Marea - chat web (demo académico).
 *
 * Seguridad: todo texto que llega del servidor se trata como no confiable y se
 * inserta únicamente con textContent/createTextNode. No se usa ninguna API que
 * interprete HTML. La CSP del servidor prohíbe además scripts y estilos inline.
 */
(function () {
  'use strict';

  const EJEMPLOS = [
    'Hola, buenas tardes',
    'Tienen lancha para 12 personas el sábado para Cholón?',
    'Cuánto me devuelven si cancelo con dos días de anticipación?',
    'Somos 8 amigos y queremos algo con música para celebrar',
    'Puedo llevar a mi perro?',
    'Cuánto cuesta ir a Playa Blanca el 10 de octubre para 6 personas?',
  ];

  const TIMEOUT_TURNO_MS = 90000;
  const SALUD_RAPIDA_MS = 15000;
  const SALUD_LENTA_MS = 60000;
  const MAX_TEXTO_CHIP = 160;
  const MAX_TEXTO_MENSAJE = 20000;
  const MAX_COMENTARIO = 500;
  const CLAVE_CONSENTIMIENTO = 'vm.consentimiento';

  const MENSAJES_ERROR = {
    entrada_invalida: 'No pude procesar ese mensaje. Revisa que no esté vacío ni sea demasiado largo.',
    sesion_desconocida: 'Se perdió la sesión. Empieza una conversación nueva.',
    limite_sesion: 'Se acabaron los turnos de esta conversación.',
    limite_ritmo: 'Vas muy rápido. Espera unos segundos antes de volver a escribir.',
    tope_diario: 'El demo llegó a su tope de uso de hoy. Vuelve a intentarlo mañana.',
    saturado: 'El servicio está ocupado en este momento. Puedes reintentar en unos segundos.',
    iniciando: 'El servicio se está iniciando. Reintenta en unos segundos.',
    error_interno: 'Ocurrió un problema de nuestro lado. Puedes reintentar.',
  };

  const peso = new Intl.NumberFormat('es-CO', {
    style: 'currency',
    currency: 'COP',
    maximumFractionDigits: 0,
  });

  const estado = {
    sessionId: null,
    turnosRestantes: null,
    maxCaracteres: 1000,
    ttlReservaMin: 30,
    registrarTexto: false,
    ocupado: false,
    bloqueado: false, // sin turnos o tope diario: no se puede enviar
    pendiente: null, // {texto, requestId} del turno que se puede reintentar
    epoca: 0, // se incrementa al empezar de nuevo para ignorar respuestas viejas
    pegado: true, // el usuario está al final del chat
    controlTurno: null,
    temporizadorBanner: null,
    temporizadorSalud: null,
  };

  const $ = (id) => document.getElementById(id);
  const ui = {};

  /* ---------- utilidades DOM ---------- */

  function h(tag, opciones, ...hijos) {
    const nodo = document.createElement(tag);
    const o = opciones || {};
    if (o.clase) nodo.className = o.clase;
    if (o.texto != null) nodo.textContent = String(o.texto);
    if (o.attrs) {
      for (const [k, v] of Object.entries(o.attrs)) {
        if (/^on/i.test(k) || k.toLowerCase() === 'style') continue;
        nodo.setAttribute(k, String(v));
      }
    }
    if (o.on) {
      for (const [ev, fn] of Object.entries(o.on)) nodo.addEventListener(ev, fn);
    }
    for (const hijo of hijos) {
      if (hijo == null || hijo === false) continue;
      nodo.append(hijo);
    }
    return nodo;
  }

  const SVG_NS = 'http://www.w3.org/2000/svg';
  function svgPath(contenedor, trazos) {
    for (const t of trazos) {
      const p = document.createElementNS(SVG_NS, 'path');
      for (const [k, v] of Object.entries(t)) p.setAttribute(k, v);
      contenedor.append(p);
    }
  }

  function dibujarLogo() {
    svgPath(ui.logo, [
      { d: 'M4 20 L16 5 L28 20 Z', fill: 'currentColor', opacity: '0.9' },
      { d: 'M3 23 H29 L25 28 H7 Z', fill: 'currentColor' },
      { d: 'M3 30 q3.25 -2.5 6.5 0 t6.5 0 t6.5 0 t6.5 0', fill: 'none', stroke: 'currentColor', 'stroke-width': '1.6', 'stroke-linecap': 'round' },
    ]);
  }

  function texto(valor, max) {
    if (valor == null) return '';
    let s = typeof valor === 'string' ? valor : (typeof valor === 'number' || typeof valor === 'boolean') ? String(valor) : '';
    s = s.trim();
    if (max && s.length > max) s = s.slice(0, max - 1) + '…';
    return s;
  }

  function esObjeto(v) {
    return v !== null && typeof v === 'object' && !Array.isArray(v);
  }

  function lista(v) {
    return Array.isArray(v) ? v : [];
  }

  function numero(v) {
    const n = typeof v === 'number' ? v : (typeof v === 'string' && v.trim() !== '' ? Number(v) : NaN);
    return Number.isFinite(n) ? n : null;
  }

  /* ---------- red ---------- */

  async function api(ruta, { metodo = 'GET', cuerpo, timeoutMs = 20000, controlExterno } = {}) {
    const control = controlExterno || new AbortController();
    const reloj = setTimeout(() => control.abort(), timeoutMs);
    try {
      const resp = await fetch(ruta, {
        method: metodo,
        headers: cuerpo !== undefined ? { 'Content-Type': 'application/json', Accept: 'application/json' } : { Accept: 'application/json' },
        body: cuerpo !== undefined ? JSON.stringify(cuerpo) : undefined,
        signal: control.signal,
        credentials: 'same-origin',
        cache: 'no-store',
      });
      let datos = null;
      if (resp.status !== 204) {
        try { datos = await resp.json(); } catch (_) { datos = null; }
      }
      return { ok: resp.ok, status: resp.status, datos, retryAfter: resp.headers.get('Retry-After') };
    } catch (err) {
      const e = new Error('red');
      e.esRed = true;
      e.tiempoAgotado = control.signal.aborted && !(controlExterno && controlExterno.cancelado);
      throw e;
    } finally {
      clearTimeout(reloj);
    }
  }

  function errorDe(resp) {
    const e = resp && esObjeto(resp.datos) && esObjeto(resp.datos.error) ? resp.datos.error : {};
    const codigo = texto(e.codigo, 40);
    // En 5xx (salvo 503) el mensaje podría filtrar detalles internos: se usa el texto local.
    const interno = resp && resp.status >= 500 && resp.status !== 503;
    const msgServidor = interno ? '' : texto(e.mensaje, 300);
    return { codigo, mensaje: msgServidor || MENSAJES_ERROR[codigo] || (interno ? MENSAJES_ERROR.error_interno : 'Algo salió mal. Intenta de nuevo.') };
  }

  /* ---------- consentimiento ---------- */

  function leerConsentimiento() {
    try {
      const v = JSON.parse(sessionStorage.getItem(CLAVE_CONSENTIMIENTO) || 'null');
      return esObjeto(v) && v.aceptado === true ? { aceptado: true, registrar: v.registrar === true } : null;
    } catch (_) {
      return null;
    }
  }

  function guardarConsentimiento() {
    try {
      sessionStorage.setItem(CLAVE_CONSENTIMIENTO, JSON.stringify({ aceptado: true, registrar: estado.registrarTexto }));
    } catch (_) { /* almacenamiento bloqueado: se vive solo con la memoria */ }
  }

  const SELECTOR_FOCO = 'button:not([disabled]), input:not([disabled]), textarea:not([disabled]), [href], [tabindex]:not([tabindex="-1"])';

  function atraparFoco(ev) {
    if (ev.key !== 'Tab') return;
    const focos = Array.from(ui.consentDialog.querySelectorAll(SELECTOR_FOCO));
    if (!focos.length) return;
    const primero = focos[0];
    const ultimo = focos[focos.length - 1];
    const activo = document.activeElement;
    if (ev.shiftKey && (activo === primero || activo === ui.consentDialog)) {
      ev.preventDefault();
      ultimo.focus();
    } else if (!ev.shiftKey && activo === ultimo) {
      ev.preventDefault();
      primero.focus();
    } else if (!ui.consentDialog.contains(activo)) {
      ev.preventDefault();
      primero.focus();
    }
  }

  function mostrarConsentimiento() {
    return new Promise((resolver) => {
      const previo = document.activeElement;
      ui.consentCheck.checked = estado.registrarTexto;
      ui.consentOverlay.hidden = false;
      ui.app.inert = true;
      ui.consentDialog.setAttribute('tabindex', '-1');
      ui.consentDialog.focus();
      ui.consentDialog.addEventListener('keydown', atraparFoco);
      ui.consentAccept.addEventListener('click', function aceptar() {
        ui.consentAccept.removeEventListener('click', aceptar);
        ui.consentDialog.removeEventListener('keydown', atraparFoco);
        estado.registrarTexto = ui.consentCheck.checked;
        guardarConsentimiento();
        ui.consentMenu.checked = estado.registrarTexto;
        ui.consentOverlay.hidden = true;
        ui.app.inert = false;
        if (previo && previo !== document.body && document.contains(previo)) previo.focus();
        else ui.input.focus();
        resolver();
      });
    });
  }

  /* ---------- estado y salud ---------- */

  const ETIQUETAS_ESTADO = {
    listo: 'En línea',
    iniciando: 'Iniciando…',
    degradado: 'Modo degradado',
  };

  function pintarEstado(clave, etiqueta) {
    ui.status.dataset.estado = clave;
    ui.statusText.textContent = etiqueta;
    ui.degradedNotice.hidden = clave !== 'degradado';
  }

  async function consultarSalud() {
    let proximo = SALUD_RAPIDA_MS;
    try {
      const r = await api('/api/salud', { timeoutMs: 8000 });
      const d = r.ok && esObjeto(r.datos) ? r.datos : null;
      if (!d) {
        pintarEstado('error', 'Sin conexión');
      } else {
        const e = texto(d.estado, 20);
        pintarEstado(ETIQUETAS_ESTADO[e] ? e : 'desconocido', ETIQUETAS_ESTADO[e] || 'Estado desconocido');
        const ttl = numero(d.ttl_reserva_demo_min);
        if (ttl && ttl > 0) estado.ttlReservaMin = ttl;
        if (e === 'listo') proximo = SALUD_LENTA_MS;
      }
    } catch (_) {
      pintarEstado('error', 'Sin conexión');
    }
    clearTimeout(estado.temporizadorSalud);
    estado.temporizadorSalud = setTimeout(consultarSalud, proximo);
  }

  /* ---------- panel de menú y agentes ---------- */

  function alternarMenu(forzar) {
    const abrir = typeof forzar === 'boolean' ? forzar : ui.menuPanel.hidden;
    ui.menuPanel.hidden = !abrir;
    ui.menuBtn.setAttribute('aria-expanded', String(abrir));
    if (abrir) cargarAgentes();
  }

  async function cargarAgentes() {
    ui.agentsHint.hidden = false;
    ui.agentsHint.textContent = 'Cargando agentes…';
    try {
      const r = await api('/api/agentes', { timeoutMs: 8000 });
      const agentes = r.ok && esObjeto(r.datos) ? lista(r.datos.agentes) : null;
      if (!agentes) throw new Error('sin datos');
      ui.agentsList.replaceChildren(...agentes.filter(esObjeto).map(tarjetaAgente));
      ui.agentsHint.hidden = agentes.length > 0;
      if (!agentes.length) ui.agentsHint.textContent = 'No hay agentes registrados.';
    } catch (_) {
      ui.agentsList.replaceChildren();
      ui.agentsHint.textContent = 'No se pudo cargar la lista de agentes.';
    }
  }

  function tarjetaAgente(a) {
    const enLinea = a.en_linea === true;
    const nombre = texto(a.nombre, 80) || texto(a.clave, 80) || 'Agente';
    const habilidades = lista(a.habilidades).filter(esObjeto).map((hab) => h('li', { texto: texto(hab.nombre, 80) || texto(hab.id, 80) }));
    return h('li', { clase: 'agent' },
      h('div', { clase: 'agent-head' },
        h('span', { clase: 'dot ' + (enLinea ? 'dot-on' : 'dot-off'), attrs: { 'aria-hidden': 'true' } }),
        h('strong', { texto: nombre }),
        h('span', { clase: 'agent-state', texto: enLinea ? 'en línea' : 'fuera de línea' })),
      habilidades.length ? h('ul', { clase: 'skills' }, ...habilidades) : null);
  }

  /* ---------- mensajes ---------- */

  function distanciaAlFinal() {
    return ui.log.scrollHeight - ui.log.scrollTop - ui.log.clientHeight;
  }

  function alFinal() {
    ui.log.scrollTop = ui.log.scrollHeight;
    estado.pegado = true;
    ui.jumpBtn.hidden = true;
  }

  function agregar(nodo, { forzarBajar = false, mostrarInicio = false } = {}) {
    ui.log.append(nodo);
    if (!(forzarBajar || estado.pegado)) {
      ui.jumpBtn.hidden = false;
      return;
    }
    alFinal();
    // Una respuesta más alta que el visor se lee desde su primera línea, no desde el final.
    if (mostrarInicio && nodo.offsetHeight > ui.log.clientHeight) {
      const arriba = nodo.getBoundingClientRect().top - ui.log.getBoundingClientRect().top + ui.log.scrollTop;
      ui.log.scrollTop = Math.max(0, arriba - 8);
      estado.pegado = false;
      ui.jumpBtn.hidden = false;
    }
  }

  function burbujaUsuario(textoUsuario) {
    const nodo = h('article', { clase: 'msg msg-user', attrs: { 'aria-label': 'Tú' } },
      h('div', { clase: 'bubble', texto: textoUsuario }));
    agregar(nodo, { forzarBajar: true });
    return nodo;
  }

  function nota(textoNota) {
    agregar(h('div', { clase: 'system-note', texto: textoNota }));
  }

  function burbujaLocal(textoAsistente) {
    agregar(h('article', { clase: 'msg msg-bot', attrs: { 'aria-label': 'Asistente' } },
      h('div', { clase: 'bubble', texto: textoAsistente })));
  }

  function chipsCitas(citas) {
    const chips = [];
    for (const c of citas) {
      if (!esObjeto(c)) continue;
      const partes = [texto(c.titulo, MAX_TEXTO_CHIP), texto(c.seccion, MAX_TEXTO_CHIP), texto(c.version, 40)].filter(Boolean);
      const fuente = texto(c.fuente, MAX_TEXTO_CHIP);
      if (!partes.length && !fuente) continue;
      const etiqueta = partes.length ? partes.join(' · ') : fuente;
      const chip = h('li', { clase: 'cite', attrs: fuente ? { title: 'Fuente: ' + fuente } : {} },
        h('span', { texto: etiqueta }),
        fuente && partes.length ? h('span', { clase: 'sr-only', texto: ' (fuente: ' + fuente + ')' }) : null);
      chips.push(chip);
    }
    if (!chips.length) return null;
    return h('div', { clase: 'cites' },
      h('span', { clase: 'cites-label', texto: 'Fuentes' }),
      h('ul', { clase: 'cites-list' }, ...chips));
  }

  function tarjetaReserva(accion) {
    const codigo = texto(accion.codigo, 60);
    const anticipo = numero(accion.anticipo);
    return h('section', { clase: 'card-booking', attrs: { 'aria-label': 'Reserva bloqueada' } },
      h('h3', { texto: 'Reserva bloqueada (demo)' }),
      codigo ? h('p', { clase: 'booking-row' }, h('span', { texto: 'Código' }), h('code', { texto: codigo })) : null,
      anticipo != null ? h('p', { clase: 'booking-row' }, h('span', { texto: 'Anticipo' }), h('strong', { texto: peso.format(anticipo) })) : null,
      h('p', { clase: 'booking-note', texto: 'Es una reserva de prueba; se libera sola en ' + estado.ttlReservaMin + ' min.' }));
  }

  function franjaEscalamiento(contrato) {
    const motivo = texto(contrato.motivo_escalamiento, 400);
    return h('div', { clase: 'escalation', attrs: { role: 'note' } },
      h('strong', { texto: 'Este caso pasaría a una persona del equipo' }),
      motivo ? h('span', { texto: motivo }) : null);
  }

  function fila(etiqueta, valor) {
    return [h('dt', { texto: etiqueta }), h('dd', { texto: valor })];
  }

  function detalleResolucion(contrato, trazas) {
    const meta = esObjeto(contrato.metadatos) ? contrato.metadatos : {};
    const router = esObjeto(trazas.router) ? trazas.router : {};
    const filas = [];
    const intencion = texto(router.intencion, 80) || texto(meta.intencion, 80);
    if (intencion) filas.push(...fila('Intención', intencion));
    const conf = numero(router.confianza) ?? numero(meta.confianza_router);
    if (conf != null) filas.push(...fila('Confianza', Math.round(conf * 100) + ' %'));
    const metodo = texto(router.metodo, 60) || texto(meta.metodo_router, 60);
    if (metodo) filas.push(...fila('Método', metodo));
    const destino = texto(router.agente_destino, 80);
    if (destino) filas.push(...fila('Agente destino', destino));
    const consultados = lista(meta.agentes_consultados).map((x) => texto(x, 80)).filter(Boolean);
    if (consultados.length) filas.push(...fila('Agentes consultados', consultados.join(', ')));
    const msRouter = numero(router.ms);
    if (msRouter != null) filas.push(...fila('Router', Math.round(msRouter) + ' ms'));
    const msTotal = numero(meta.ms_total);
    if (msTotal != null) filas.push(...fila('Tiempo total', Math.round(msTotal) + ' ms'));
    if (meta.degradado === true) filas.push(...fila('Modo', 'degradado'));

    const saltos = lista(trazas.saltos_a2a).filter(esObjeto).map((s) => {
      const nombre = [texto(s.agente, 80), texto(s.habilidad, 80)].filter(Boolean).join('.') || 'salto';
      const ms = numero(s.ms);
      const ok = s.ok === true;
      return h('li', { clase: ok ? 'hop hop-ok' : 'hop hop-fail' },
        h('span', { texto: nombre }),
        ms != null ? h('span', { clase: 'hop-ms', texto: Math.round(ms) + ' ms' }) : null,
        h('span', { clase: 'hop-state', texto: ok ? 'ok' : 'falló' }));
    });

    const slots = esObjeto(trazas.slots) ? Object.entries(trazas.slots).filter(([, v]) => v != null && v !== '') : [];
    const slotsNodos = slots.slice(0, 20).flatMap(([k, v]) => fila(texto(k, 40), typeof v === 'object' ? JSON.stringify(v).slice(0, 120) : texto(v, 120)));

    if (!filas.length && !saltos.length && !slotsNodos.length) return null;
    return h('details', { clase: 'trace' },
      h('summary', { texto: 'Cómo se resolvió' }),
      filas.length ? h('dl', { clase: 'kv' }, ...filas) : null,
      saltos.length ? h('div', {}, h('p', { clase: 'trace-title', texto: 'Llamadas entre agentes (A2A)' }), h('ul', { clase: 'hops' }, ...saltos)) : null,
      slotsNodos.length ? h('div', {}, h('p', { clase: 'trace-title', texto: 'Datos detectados' }), h('dl', { clase: 'kv' }, ...slotsNodos)) : null);
  }

  function controlesFeedback(requestId, sessionId) {
    const caja = h('div', { clase: 'feedback' });
    let enviado = false;
    let enCurso = false;

    function mostrarError(errorTexto) {
      const previo = caja.querySelector('.feedback-error');
      if (previo) previo.remove();
      caja.append(h('span', { clase: 'feedback-error', texto: errorTexto, attrs: { role: 'alert' } }));
    }

    function pintarBase() {
      const up = h('button', { clase: 'btn btn-small', texto: 'Útil', attrs: { type: 'button' }, on: { click: () => enviar('up') } });
      const down = h('button', { clase: 'btn btn-small', texto: 'No útil', attrs: { type: 'button', 'aria-expanded': 'false' }, on: { click: () => abrirComentario(down) } });
      caja.replaceChildren(
        h('span', { clase: 'feedback-label', texto: '¿Te sirvió?' }), up, down);
    }

    function abrirComentario(botonDown) {
      botonDown.setAttribute('aria-expanded', 'true');
      const area = h('textarea', { attrs: { rows: 2, maxlength: MAX_COMENTARIO, id: 'fb-' + requestId, placeholder: 'Cuéntanos qué faltó (opcional)' } });
      const etiqueta = h('label', { clase: 'sr-only', texto: 'Comentario opcional', attrs: { for: 'fb-' + requestId } });
      const enviarBtn = h('button', { clase: 'btn btn-small btn-primary', texto: 'Enviar', attrs: { type: 'button' }, on: { click: () => enviar('down', area.value) } });
      const form = h('div', { clase: 'feedback-form' }, etiqueta, area, enviarBtn);
      caja.append(form);
      area.focus();
    }

    async function enviar(valor, comentario) {
      if (enviado || enCurso) return;
      enCurso = true;
      caja.querySelectorAll('button, textarea').forEach((n) => { n.disabled = true; });
      const cuerpo = { session_id: sessionId, request_id: requestId, valor };
      const c = texto(comentario, MAX_COMENTARIO);
      if (c) cuerpo.comentario = c;
      let fallo = false;
      try {
        const r = await api('/api/feedback', { metodo: 'POST', cuerpo, timeoutMs: 15000 });
        fallo = !r.ok;
      } catch (_) {
        fallo = true;
      }
      enCurso = false;
      if (fallo) {
        // Se conserva el formulario para no perder el comentario escrito.
        caja.querySelectorAll('button, textarea').forEach((n) => { n.disabled = false; });
        mostrarError('No se pudo enviar tu valoración. Intenta de nuevo.');
        return;
      }
      enviado = true;
      caja.replaceChildren(h('span', { clase: 'feedback-thanks', texto: 'Gracias', attrs: { role: 'status' } }));
    }

    pintarBase();
    return caja;
  }

  function burbujaAsistente(respuesta, sessionId) {
    const contrato = esObjeto(respuesta.contrato) ? respuesta.contrato : {};
    const trazas = esObjeto(respuesta.trazas) ? respuesta.trazas : {};
    const requestId = texto(respuesta.request_id, 80);
    const mensaje = texto(contrato.mensaje, MAX_TEXTO_MENSAJE) || 'No obtuve una respuesta para mostrar. Intenta reformular tu mensaje.';
    const acciones = lista(contrato.acciones).filter(esObjeto);
    const reserva = acciones.find((a) => a.tipo === 'reserva_bloqueada');
    const escala = contrato.escalar_a_humano === true || contrato.tipo === 'escalamiento';

    const nodo = h('article', { clase: 'msg msg-bot', attrs: { 'aria-label': 'Asistente' } },
      h('div', { clase: 'bubble' },
        h('div', { clase: 'bubble-text', texto: mensaje }),
        escala ? franjaEscalamiento(contrato) : null,
        reserva ? tarjetaReserva(reserva) : null,
        chipsCitas(lista(contrato.citas)),
        detalleResolucion(contrato, trazas)),
      requestId ? controlesFeedback(requestId, sessionId) : null);
    agregar(nodo, { mostrarInicio: true });
  }

  /* ---------- banner de errores ---------- */

  function ocultarBanner() {
    clearInterval(estado.temporizadorBanner);
    ui.banner.hidden = true;
    ui.bannerAction.hidden = true;
    ui.bannerAction.onclick = null;
  }

  function mostrarBanner(mensaje, { etiqueta, alPulsar, esperaSeg = 0 } = {}) {
    ocultarBanner();
    ui.bannerText.textContent = mensaje;
    ui.banner.hidden = false;
    if (!etiqueta) return;
    ui.bannerAction.hidden = false;
    ui.bannerAction.onclick = alPulsar;
    let restante = esperaSeg;
    const pintar = () => {
      ui.bannerAction.disabled = restante > 0;
      ui.bannerAction.textContent = restante > 0 ? etiqueta + ' (' + restante + ' s)' : etiqueta;
    };
    pintar();
    if (restante > 0) {
      estado.temporizadorBanner = setInterval(() => {
        restante -= 1;
        pintar();
        if (restante <= 0) clearInterval(estado.temporizadorBanner);
      }, 1000);
    }
  }

  /* ---------- sesión y turnos ---------- */

  function actualizarTurnos(n) {
    const v = numero(n);
    if (v == null) return;
    estado.turnosRestantes = Math.max(0, Math.floor(v));
    ui.turns.textContent = 'Turnos restantes: ' + estado.turnosRestantes;
    ui.turns.classList.toggle('turns-low', estado.turnosRestantes <= 3);
    if (estado.turnosRestantes === 0) bloquearPorTurnos();
  }

  function bloquearPorTurnos() {
    estado.bloqueado = true;
    mostrarBanner('Llegaste al final de esta conversación: no quedan turnos.', {
      etiqueta: 'Empezar una conversación nueva',
      alPulsar: reiniciarConversacion,
    });
    refrescarControles();
  }

  async function crearSesion() {
    const r = await api('/api/sesion', { metodo: 'POST', cuerpo: {}, timeoutMs: 15000 });
    if (!r.ok || !esObjeto(r.datos) || !texto(r.datos.session_id)) {
      const e = new Error('sesion');
      e.respuesta = r;
      throw e;
    }
    const d = r.datos;
    estado.sessionId = texto(d.session_id, 200);
    const max = numero(d.max_caracteres);
    if (max && max > 0) {
      estado.maxCaracteres = Math.floor(max);
      ui.input.maxLength = estado.maxCaracteres;
    }
    const ttl = numero(d.ttl_reserva_demo_min);
    if (ttl && ttl > 0) estado.ttlReservaMin = ttl;
    actualizarTurnos(d.turnos_restantes);
    actualizarContador();
  }

  function refrescarControles() {
    const sinTexto = ui.input.value.trim() === '';
    ui.sendBtn.disabled = estado.ocupado || estado.bloqueado || sinTexto;
    ui.input.disabled = estado.bloqueado;
    ui.examples.querySelectorAll('button').forEach((b) => { b.disabled = estado.ocupado || estado.bloqueado; });
    ui.typing.hidden = !estado.ocupado;
    ui.composerWrap.setAttribute('aria-busy', String(estado.ocupado));
  }

  function actualizarContador() {
    ui.counter.textContent = ui.input.value.length + ' / ' + estado.maxCaracteres;
  }

  function ajustarAltura() {
    ui.input.style.height = 'auto';
    ui.input.style.height = Math.min(ui.input.scrollHeight, 140) + 'px';
  }

  function enviarMensaje(textoUsuario) {
    const t = textoUsuario.trim();
    if (!t || estado.ocupado || estado.bloqueado) return;
    ocultarBanner();
    burbujaUsuario(t);
    ui.input.value = '';
    ajustarAltura();
    actualizarContador();
    plegarEjemplos(true);
    estado.pendiente = { texto: t, requestId: crypto.randomUUID() };
    ejecutarTurno();
  }

  function reintentar() {
    if (estado.ocupado || !estado.pendiente) return;
    ocultarBanner();
    ejecutarTurno();
  }

  async function ejecutarTurno() {
    const epoca = estado.epoca;
    const { texto: t, requestId } = estado.pendiente;
    estado.ocupado = true;
    refrescarControles();
    let sesionRecreada = false;

    try {
      for (;;) {
        if (!estado.sessionId) {
          try {
            await crearSesion();
          } catch (e) {
            if (epoca !== estado.epoca) return;
            if (e.esRed) return falloRed(e);
            return falloServidor(e.respuesta);
          }
        }
        const control = new AbortController();
        estado.controlTurno = control;
        let r;
        try {
          r = await api('/api/turno', {
            metodo: 'POST',
            timeoutMs: TIMEOUT_TURNO_MS,
            controlExterno: control,
            cuerpo: {
              session_id: estado.sessionId,
              texto: t,
              request_id: requestId,
              consentimiento_registro: estado.registrarTexto,
            },
          });
        } catch (e) {
          if (epoca !== estado.epoca) return;
          return falloRed(e);
        }
        if (epoca !== estado.epoca) return;

        if (r.status === 404 && errorDe(r).codigo === 'sesion_desconocida' && !sesionRecreada) {
          sesionRecreada = true;
          estado.sessionId = null;
          nota('Se perdió el contexto de la conversación anterior. Seguimos en una sesión nueva.');
          continue;
        }
        if (r.ok && esObjeto(r.datos)) {
          estado.pendiente = null;
          burbujaAsistente(r.datos, estado.sessionId);
          if (esObjeto(r.datos.sesion)) actualizarTurnos(r.datos.sesion.turnos_restantes);
          return;
        }
        return falloServidor(r);
      }
    } finally {
      if (epoca === estado.epoca) {
        estado.ocupado = false;
        estado.controlTurno = null;
        refrescarControles();
        if (!estado.bloqueado) ui.input.focus({ preventScroll: true });
      }
    }
  }

  function falloRed(e) {
    const msg = e && e.tiempoAgotado
      ? 'La respuesta está tardando demasiado. Puedes reintentar: no se duplicará tu mensaje.'
      : 'No se pudo conectar con el servidor. Revisa tu conexión y reintenta: no se duplicará tu mensaje.';
    mostrarBanner(msg, { etiqueta: 'Reintentar', alPulsar: reintentar });
  }

  function falloServidor(r) {
    const { codigo, mensaje } = errorDe(r);
    const espera = Math.min(300, Math.max(1, parseInt(r && r.retryAfter, 10) || 5));

    switch (codigo) {
      case 'limite_sesion':
        actualizarTurnos(0);
        estado.pendiente = null;
        mostrarBanner(mensaje, { etiqueta: 'Empezar una conversación nueva', alPulsar: reiniciarConversacion });
        estado.bloqueado = true;
        return;
      case 'tope_diario':
        estado.bloqueado = true;
        mostrarBanner(mensaje);
        return;
      case 'limite_ritmo':
        mostrarBanner(mensaje, { etiqueta: 'Reintentar', alPulsar: reintentar, esperaSeg: espera });
        return;
      case 'entrada_invalida':
        ui.input.value = estado.pendiente ? estado.pendiente.texto : '';
        estado.pendiente = null;
        ajustarAltura();
        actualizarContador();
        mostrarBanner(mensaje);
        return;
      case 'saturado':
      case 'iniciando':
        mostrarBanner(mensaje, { etiqueta: 'Reintentar', alPulsar: reintentar, esperaSeg: r && r.retryAfter ? espera : 0 });
        return;
      case 'sesion_desconocida':
        estado.sessionId = null;
        mostrarBanner(mensaje, { etiqueta: 'Reintentar', alPulsar: reintentar });
        return;
      default:
        mostrarBanner(mensaje, { etiqueta: 'Reintentar', alPulsar: reintentar });
    }
  }

  function reiniciarConversacion() {
    estado.epoca += 1;
    if (estado.controlTurno) {
      estado.controlTurno.cancelado = true;
      estado.controlTurno.abort();
    }
    estado.controlTurno = null;
    estado.ocupado = false;
    estado.bloqueado = false;
    estado.pendiente = null;
    estado.sessionId = null;
    estado.turnosRestantes = null;
    ui.turns.textContent = 'Turnos restantes: —';
    ui.turns.classList.remove('turns-low');
    ocultarBanner();
    ui.log.replaceChildren();
    estado.pegado = true;
    ui.jumpBtn.hidden = true;
    ui.input.value = '';
    ajustarAltura();
    actualizarContador();
    plegarEjemplos(false);
    alternarMenu(false);
    saludoInicial();
    refrescarControles();
    crearSesion().catch(() => { /* se reintenta al enviar el primer mensaje */ });
    ui.input.focus();
  }

  function saludoInicial() {
    burbujaLocal('¡Hola! Soy el asistente de Vallis Marea. Puedo ayudarte con lanchas en Cartagena: disponibilidad, precios, políticas y recomendaciones. Todo es de prueba. ¿En qué te ayudo?');
  }

  /* ---------- ejemplos ---------- */

  function plegarEjemplos(plegar) {
    ui.examplesList.hidden = plegar;
    ui.examplesToggle.setAttribute('aria-expanded', String(!plegar));
  }

  function montarEjemplos() {
    ui.examples.replaceChildren(...EJEMPLOS.map((t) => h('button', {
      clase: 'chip', texto: t, attrs: { type: 'button' }, on: { click: () => enviarMensaje(t) },
    })));
  }

  /* ---------- arranque ---------- */

  function enlazarEventos() {
    ui.composer.addEventListener('submit', (ev) => {
      ev.preventDefault();
      enviarMensaje(ui.input.value);
    });
    ui.input.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing) {
        ev.preventDefault();
        enviarMensaje(ui.input.value);
      }
    });
    ui.input.addEventListener('input', () => {
      actualizarContador();
      ajustarAltura();
      refrescarControles();
    });
    ui.log.addEventListener('scroll', () => {
      estado.pegado = distanciaAlFinal() < 80;
      if (estado.pegado) ui.jumpBtn.hidden = true;
    }, { passive: true });
    ui.jumpBtn.addEventListener('click', alFinal);
    ui.examplesToggle.addEventListener('click', () => plegarEjemplos(!ui.examplesList.hidden));
    ui.menuBtn.addEventListener('click', () => alternarMenu());
    ui.newChatMenu.addEventListener('click', reiniciarConversacion);
    ui.consentMenu.addEventListener('change', () => {
      estado.registrarTexto = ui.consentMenu.checked;
      guardarConsentimiento();
    });
    document.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape' && !ui.menuPanel.hidden) {
        alternarMenu(false);
        ui.menuBtn.focus();
      }
    });
  }

  async function iniciar() {
    const nombres = {
      app: 'app', logo: 'brand-logo', status: 'status', statusText: 'status-text', degradedNotice: 'degraded-notice',
      menuBtn: 'menu-btn', menuPanel: 'menu-panel', consentMenu: 'consent-menu', agentsHint: 'agents-hint',
      agentsList: 'agents-list', newChatMenu: 'new-chat-menu', log: 'log', jumpBtn: 'jump-btn', typing: 'typing',
      banner: 'banner', bannerText: 'banner-text', bannerAction: 'banner-action', examples: 'examples',
      examplesList: 'examples', examplesToggle: 'examples-toggle', composer: 'composer', composerWrap: null,
      input: 'input', sendBtn: 'send-btn', counter: 'counter', turns: 'turns',
      consentOverlay: 'consent-overlay', consentDialog: 'consent-dialog', consentCheck: 'consent-check',
      consentAccept: 'consent-accept',
    };
    for (const [k, id] of Object.entries(nombres)) if (id) ui[k] = $(id);
    ui.composerWrap = document.querySelector('.composer-wrap');

    dibujarLogo();
    montarEjemplos();
    enlazarEventos();
    pintarEstado('desconocido', 'Conectando…');
    consultarSalud();
    saludoInicial();
    refrescarControles();

    const previo = leerConsentimiento();
    if (previo) {
      estado.registrarTexto = previo.registrar;
      ui.consentMenu.checked = previo.registrar;
    } else {
      await mostrarConsentimiento();
    }
    try {
      await crearSesion();
    } catch (_) { /* se reintenta al enviar el primer mensaje */ }
    refrescarControles();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', iniciar);
  else iniciar();
})();
