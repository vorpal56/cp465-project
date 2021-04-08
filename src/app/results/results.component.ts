import { Component, OnInit, Input, Output, EventEmitter } from '@angular/core';
import { Observable } from 'rxjs';

@Component({
  selector: 'search-results',
  templateUrl: './results.component.html',
  styleUrls: ['./results.component.scss']
})
export class ResultsComponent implements OnInit {
	@Input("results") searchResults$: Observable<any>;

  constructor() { }

  ngOnInit(): void { }

	partialContent(content: string) {
		return content.slice(0, 50) + "..."
	}
}
